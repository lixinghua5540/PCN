import torch
import torch.nn as nn
import numpy as np
import functools
import torch.nn.functional as F
from rsiseg.models.decode_heads.segformer_head import SegformerHead
from rsiseg.ops import resize

class MultiModelEnsemble(nn.Module):
    def __init__(self, n_models, n_classes):
        super().__init__()
        self.weights = nn.Parameter(torch.zeros(n_models, n_classes))
        
    def forward(self, preds):
        weights = torch.softmax(self.weights, dim=0)
        return torch.einsum('mc,mbchw->bchw', weights, preds)
    
class DWSC_unit(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, groups):
        super(DWSC_unit, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            groups=groups,
            bias=False
        )

    def forward(self, input):
        out = self.conv(input)
        return out
class DWSC(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, padding, norm):
        super(DWSC, self).__init__()
        self.depthwiseconv = DWSC_unit(in_ch,in_ch,kernel_size,padding=padding,groups=in_ch)#
        self.pointwiseconv = DWSC_unit(in_ch,out_ch,kernel_size=1,padding=0,groups=1)#
        self.relu1=nn.PReLU()
        self.relu2=nn.PReLU()

        if norm=="BatchNorm2d":
            self.norm1=nn.BatchNorm2d(num_features=in_ch)#
            #print(self.norm1)
            self.norm2=nn.BatchNorm2d(num_features=out_ch)#
        else:
            self.norm1=nn.InstanceNorm2d(num_features=in_ch)#
            self.norm2=nn.InstanceNorm2d(num_features=in_ch)#



        #self.process=nn.sequential()
    def forward(self, input):
        dwr=self.depthwiseconv(input)
        dwr=self.relu1(self.norm1(dwr))
        pwr=self.pointwiseconv(dwr)
        #out = self.relu2(self.norm2(pwr))
        return pwr
    
class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, norm, upsample=None):
    #def __init__(self, in_channels, out_channels, kernel_size, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(scale_factor=upsample,mode="bilinear",align_corners=True)
        self.reflection_padding = int(np.floor(kernel_size / 2))
        if self.reflection_padding != 0:
            self.reflection_pad = nn.ReflectionPad2d(self.reflection_padding)
        #self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)
        self.conv2d=DWSC(in_channels,out_channels,kernel_size,padding=0, norm="BatchNorm2d")#??
        #self.conv2d=AtrousSeparableConvolution(in_channels,out_channels,kernel_size,stride=1,padding=0)

    def forward(self, x):
        if self.upsample:
            x = self.upsample_layer(x)
        if self.reflection_padding != 0:
            x = self.reflection_pad(x)
        out = self.conv2d(x)
        return out
    
class SpatialAttentionMoudle(nn.Module):
    def __init__(self):
        super(SpatialAttentionMoudle, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class ChannelAttentionMoudle(nn.Module):
    def __init__(self, channel, ratio=2):
        super(ChannelAttentionMoudle, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel//ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionMoudle(channel)
        self.spatial_attention = SpatialAttentionMoudle()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out
    
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        #print(norm_layer)
        #test_norm=norm_layer(512)
        #print(test_norm)
        conv_block += [#nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                                DWSC(dim,dim,kernel_size=3,padding=p,norm="BatchNorm2d"),#??
                                #AtrousSeparableConvolution(dim,dim,kernel_size=3,padding=p),
                                norm_layer(dim), nn.LeakyReLU(0.1,True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [#nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), 
                                DWSC(dim,dim,kernel_size=3,padding=p,norm="BatchNorm2d"),#??
                                #AtrousSeparableConvolution(dim,dim,kernel_size=3,padding=p),
                                norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

        

class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    #def __init__(self, input_nc, kw, padw, ndf=64, n_layers=3, norm_layer="BatchNorm2d"):
    def __init__(self, input_nc, kw, padw, ndf=64, n_layers=3, norm_layer="BatchNorm2d"):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if norm_layer=="BatchNorm2d":
            norm_layer = nn.BatchNorm2d
        else:
            norm_layer = nn.InstanceNorm2d

        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        #input_nc=3
        #kw = 4
        #padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class GANLoss(nn.Module):
    """Define different GAN objectives.

    The GANLoss class abstracts away the need to create the target label tensor
    that has the same size as the input.
    """

    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        """ Initialize the GANLoss class.

        Parameters:
            gan_mode (str) - - the type of GAN objective. It currently supports vanilla, lsgan, and wgangp.
            target_real_label (bool) - - label for a real image
            target_fake_label (bool) - - label of a fake image

        Note: Do not use sigmoid as the last layer of Discriminator.
        LSGAN needs no sigmoid. vanilla GANs will handle it with BCEWithLogitsLoss.
        """
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        """Create label tensors with the same size as the input.

        Parameters:
            prediction (tensor) - - tpyically the prediction from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            A label tensor filled with ground truth label, and with the size of the input
        """

        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        """Calculate loss given Discriminator's output and grount truth labels.

        Parameters:
            prediction (tensor) - - tpyically the prediction output from a discriminator
            target_is_real (bool) - - if the ground truth label is for real images or fake images

        Returns:
            the calculated loss.
        """
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def compute_discriminator_loss(fake_images, seg_masks, discriminator):#is necessary to input a mask information it is effective for specific class?
    fake_loss= 0.0
    #seg_masks = seg_masks.long()
    _, num_classes, _, _ = fake_images.shape
    _, _, H_out, W_out = discriminator(fake_images).shape
    for i in range(num_classes):
        inds=(seg_masks==i).int()
        inds_down = F.interpolate(inds.float(), size=(H_out, W_out), mode='bilinear')#the format
        activate_img=fake_images*inds#broadcast
        activate_z=discriminator(activate_img)
        fake_loss_class = torch.mean(torch.square(activate_z-torch.ones_like(activate_z))*inds_down)
        fake_loss += fake_loss_class

    return fake_loss

def compute_discriminator_loss2(real_images, fake_images, seg_masks, fake_masks, discriminator):
    loss= 0.0
    #seg_masks = seg_masks.long()
    #fake_masks
    _, num_classes, _, _ = real_images.shape
    _, _, H_out, W_out = discriminator(real_images).shape
    for i in range(num_classes):
        inds=(seg_masks==i).int()
        inds_down = F.interpolate(inds.float(), size=(H_out, W_out), mode='bilinear')#the format
        indsf=(fake_masks==i).int()
        indsf_down = F.interpolate(indsf.float(), size=(H_out, W_out), mode='bilinear')#the format
        activate_img=real_images*inds#broadcast
        activate_z=discriminator(activate_img)
        real_loss_class = torch.mean(torch.square(activate_z-torch.ones_like(activate_z))*inds_down)
        activatef_img=fake_images*indsf
        activatef_z=discriminator(activatef_img)
        fake_loss_class = torch.mean(torch.square(activatef_z-torch.zeros_like(activatef_z))*indsf_down)
        loss += (real_loss_class+fake_loss_class)

    return loss


def compute_discriminator_loss3(fake_images, seg_masks, discriminator):
    fake_loss= 0.0
    num_classes = discriminator.num_classes
    _, _, H_out, W_out = discriminator(fake_images)[0].shape
    for i in range(num_classes):
        inds=(seg_masks==i).int()
        inds_down = F.interpolate(inds.float(), size=(H_out, W_out), mode='bilinear')#the format
        activate_img=fake_images*inds#broadcast
        activate_z=discriminator(activate_img)[i]
        fake_loss_class = torch.mean(torch.square(activate_z-torch.ones_like(activate_z))*inds_down)
        fake_loss += fake_loss_class

    return fake_loss

def compute_discriminator_loss4(real_images, fake_images, seg_masks, fake_masks, discriminator):
    loss= 0.0
    num_classes = discriminator.num_classes
    _, _, H_out, W_out = discriminator(real_images)[0].shape
    for i in range(num_classes):
        inds=(seg_masks==i).int()
        inds_down = F.interpolate(inds.float(), size=(H_out, W_out), mode='bilinear')#the format
        indsf=(fake_masks==i).int()
        indsf_down = F.interpolate(indsf.float(), size=(H_out, W_out), mode='bilinear')#the format
        activate_img=real_images*inds#broadcast
        activate_z=discriminator(activate_img)[i]
        real_loss_class = torch.mean(torch.square(activate_z-torch.ones_like(activate_z))*inds_down)
        activatef_img=fake_images*indsf
        activatef_z=discriminator(activatef_img)[i]
        fake_loss_class = torch.mean(torch.square(activatef_z-torch.zeros_like(activatef_z))*indsf_down)
        loss += (real_loss_class+fake_loss_class)

    return loss



def compute_discriminator_loss5(fake_images, seg_masks, discriminator):
    """Compute adversarial loss for fake images only with sparse categorical crossentropy"""
    # Get discriminator outputs (after softmax)
    fake_output = discriminator(fake_images)
    
    # Prepare target tensor (class indices)
    target = seg_masks.squeeze(1).long()  # [B, H, W]
    
    # Resize target to match discriminator output size
    _, _, H_out, W_out = fake_output.shape
    target = F.interpolate(target.unsqueeze(1).float(), 
                         size=(H_out, W_out), mode='nearest').squeeze(1).long()
    
    # Compute sparse categorical crossentropy loss
    return F.cross_entropy(fake_output, target, reduction='mean')

def compute_discriminator_loss6(real_images, fake_images, seg_masks, fake_masks, discriminator):
    """Compute adversarial loss with sparse categorical crossentropy"""
    loss = 0.0
    num_classes = discriminator.num_classes
    
    # Get discriminator outputs (after softmax)
    real_output = discriminator(real_images)
    fake_output = discriminator(fake_images)
    
    # Prepare target tensors (class indices)
    real_target = seg_masks.squeeze(1).long()  # [B, H, W]
    fake_target = fake_masks.squeeze(1).long()
    
    # Resize targets to match discriminator output size
    _, _, H_out, W_out = real_output.shape
    real_target = F.interpolate(real_target.unsqueeze(1).float(), 
                              size=(H_out, W_out), mode='nearest').squeeze(1).long()
    fake_target = F.interpolate(fake_target.unsqueeze(1).float(),
                               size=(H_out, W_out), mode='nearest').squeeze(1).long()
    
    # Compute sparse categorical crossentropy loss
    real_loss = F.cross_entropy(real_output, real_target, reduction='mean')
    fake_loss = F.cross_entropy(fake_output, fake_target, reduction='mean')
    
    return real_loss + fake_loss
        


class NewUnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_classes, patch_size, ngf=64, norm_layer="BatchNorm2d", with_semantic=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        A sequential process of UNet.
        """
        super(NewUnetGenerator, self).__init__()
        if norm_layer=="BatchNorm2d":
            self.norm_layer=nn.BatchNorm2d
        else:
            self.norm_layer=nn.InstanceNorm2d
        if type(self.norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        # if input_nc is None:
        #     input_nc = outer_nc
        print(norm_layer)
        self.num_classes=num_classes
        self.patch_size=patch_size
        self.with_semantic = with_semantic
        self.downconv = nn.Conv2d(input_nc, ngf, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        self.downnorm = self.norm_layer(ngf)
        self.last_up_conv = UpsampleConvLayer(ngf * 2, ngf * 2,kernel_size=5,stride=1,upsample=2,norm=norm_layer)
        #self.outconv = nn.Conv2d(ngf * 2, output_nc,kernel_size=5,stride=1,norm=norm_layer)
        self.outconv = nn.Conv2d(ngf * 2, output_nc, kernel_size=5,stride=1, padding=2, bias=use_bias)
        self.CBAM1=CBAM(ngf)#is this necessary?
        #downconvouter = nn.Conv2d(input_nc, inner_nc, kernel_size=7,#kernel=4,stride=2
        #                     stride=2, padding=3, bias=use_bias)
        self.downrelu = nn.LeakyReLU(0.1, True)
        self.uprelu = nn.LeakyReLU(0.1, True)
        self.downconv1 = nn.Conv2d(ngf, ngf * 2, kernel_size=4,stride=2, padding=1, bias=use_bias)
        self.downconv2 = nn.Conv2d(ngf * 2, ngf * 4, kernel_size=4,stride=2, padding=1, bias=use_bias)
        self.downconv3 = nn.Conv2d(ngf * 4, ngf * 8, kernel_size=4,stride=2, padding=1, bias=use_bias)
        self.downconv4 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,stride=2, padding=1, bias=use_bias)
        self.downconv5 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,stride=2, padding=1, bias=use_bias)
        self.downconv6 = nn.Conv2d(ngf * 8, ngf * 8, kernel_size=4,stride=2, padding=1, bias=use_bias)#4×4 it does not contain enough information for segmentation
        self.downnorm1 = self.norm_layer(ngf * 2)#
        self.downnorm2 = self.norm_layer(ngf * 4)#
        self.downnorm3 = self.norm_layer(ngf * 8)#
        self.downnorm4 = self.norm_layer(ngf * 8)#
        self.downnorm5 = self.norm_layer(ngf * 8)#
        #self.downnorm4 = self.norm_layer(ngf * 8)#
        self.upnorm1 = self.norm_layer(ngf * 8)
        self.upnorm2 = self.norm_layer(ngf * 8)
        self.upnorm3 = self.norm_layer(ngf * 8)
        self.upnorm4 = self.norm_layer(ngf * 4)
        self.upnorm5 = self.norm_layer(ngf * 2)
        self.upnorm6 = self.norm_layer(ngf * 1)
        self.upconv1=UpsampleConvLayer(ngf * 8 , ngf * 8, kernel_size=5,stride=1,upsample=2,norm=norm_layer)
        self.upconv2=UpsampleConvLayer(ngf * 16 , ngf * 8, kernel_size=5,stride=1,upsample=2,norm=norm_layer)
        self.upconv3=UpsampleConvLayer(ngf * 16 , ngf * 8, kernel_size=5,stride=1,upsample=2,norm=norm_layer)
        self.upconv4=UpsampleConvLayer(ngf * 8 * 2, ngf * 4, kernel_size=5,stride=1,upsample=2,norm=norm_layer)
        self.upconv5=UpsampleConvLayer(ngf * 4 * 2, ngf * 2, kernel_size=5,stride=1,upsample=2,norm=norm_layer)
        self.upconv6=UpsampleConvLayer(ngf * 4, ngf, kernel_size=5,stride=1,upsample=2,norm=norm_layer)
        
        self.res1=ResnetBlock( ngf*2 , padding_type='reflect', norm_layer=self.norm_layer, use_dropout=False , use_bias=use_bias)
        self.res2=ResnetBlock( ngf * 4, padding_type='reflect', norm_layer=self.norm_layer, use_dropout=False , use_bias=use_bias)
        self.res3=ResnetBlock( ngf * 8, padding_type='reflect', norm_layer=self.norm_layer, use_dropout=False , use_bias=use_bias)
        self.res4=ResnetBlock( ngf * 8, padding_type='reflect', norm_layer=self.norm_layer, use_dropout=False , use_bias=use_bias)
        self.res5=ResnetBlock( ngf * 8, padding_type='reflect', norm_layer=self.norm_layer, use_dropout=False , use_bias=use_bias)
        self.tanh=nn.Tanh()
        if self.with_semantic==True:
            self.s_head=SegformerHead(in_channels=(128,256,512,512,1024,1024,512,256), in_index=[0,1,2,3,4,5,6,7], channels=1024, num_classes=num_classes)

    def forward(self, input):
        """Standard forward"""
        x=self.downnorm(self.downconv(input))
        x1=self.downnorm1(self.downconv1(self.downrelu(self.CBAM1(x))))#is downrelu sharing strategy suitable
        x2=self.downnorm2(self.downconv2(self.downrelu(self.res1(x1))))
        x3=self.downnorm3(self.downconv3(self.downrelu(self.res2(x2))))
        x4=self.downnorm4(self.downconv4(self.downrelu(self.res3(x3))))
        x5=self.downnorm5(self.downconv5(self.downrelu(self.res4(x4))))
        x6=self.downconv6(self.downrelu(self.res5(x5)))
        x6_up=self.upnorm1(self.upconv1(self.uprelu(x6)))
        x_c5=torch.cat([x5, x6_up], 1)
        x5_up=self.upnorm2(self.upconv2(self.uprelu(x_c5)))
        x_c4=torch.cat([x4, x5_up], 1)
        x4_up=self.upnorm3(self.upconv3(self.uprelu(x_c4)))
        x_c3=torch.cat([x3, x4_up], 1)
        x3_up=self.upnorm4(self.upconv4(self.uprelu(x_c3)))
        x_c2=torch.cat([x2, x3_up], 1)
        x2_up=self.upnorm5(self.upconv5(self.uprelu(x_c2)))
        x_c1=torch.cat([x1, x2_up], 1)
        x1_up=self.upnorm6(self.upconv6(self.uprelu(x_c1)))
        x_c=torch.cat([x, x1_up], 1)
        x_lc = self.last_up_conv(self.uprelu(x_c))
        x_up=3*self.tanh(self.outconv(self.uprelu(x_lc)))
        if self.with_semantic==True:

            semantic=self.s_head([x1,x2,x3,x4,x_c4,x_c3,x_c2,x_c1])
            semantic = resize(
                semantic,
                size=[self.patch_size,self.patch_size],
                mode='bilinear',
                align_corners=True,
                warning=False)

            return x_up, semantic, x6
        else:
            return x_up