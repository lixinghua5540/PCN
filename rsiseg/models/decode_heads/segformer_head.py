# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule

from rsiseg.models.builder import HEADS
from rsiseg.models.decode_heads.decode_head import BaseDecodeHead
from rsiseg.ops import resize


@HEADS.register_module()
class SegformerHead(BaseDecodeHead):
    """The all mlp Head of segformer.

    This head is the implementation of
    `Segformer <https://arxiv.org/abs/2105.15203>` _.

    Args:
        interpolate_mode: The interpolate mode of MLP head upsample operation.
            Default: 'bilinear'.
    """

    def __init__(self, interpolate_mode='bilinear', **kwargs):
        super().__init__(input_transform='multiple_select', **kwargs)

        self.interpolate_mode = interpolate_mode
        num_inputs = len(self.in_channels)

        assert num_inputs == len(self.in_index)

        self.convs = nn.ModuleList()
        for i in range(num_inputs):
            self.convs.append(
                ConvModule(
                    in_channels=self.in_channels[i],
                    out_channels=self.channels,
                    kernel_size=1,
                    stride=1,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

        self.fusion_conv = ConvModule(
            in_channels=self.channels * num_inputs,
            out_channels=self.channels,
            kernel_size=1,
            norm_cfg=self.norm_cfg)

    def forward(self, inputs, return_features=False):
        # Receive 4 stage backbone feature map: 1/4, 1/8, 1/16, 1/32
        inputs = self._transform_inputs(inputs)
        outs = []
        for idx in range(len(inputs)):
            x = inputs[idx]
            conv = self.convs[idx]
            outs.append(
                resize(
                    input=conv(x),
                    size=inputs[0].shape[2:],
                    mode=self.interpolate_mode,
                    align_corners=self.align_corners))

        out = self.fusion_conv(torch.cat(outs, dim=1))
        features = out.clone()
        out = self.cls_seg(out)
        if return_features:
            out = (out, features)
        return out

class MLP(nn.Module):
    """Linear Embedding."""

    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2).contiguous()
        x = self.proj(x)
        return x


@HEADS.register_module()
class SegFormerHead(BaseDecodeHead):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with
    Transformers
    """

    def __init__(self, **kwargs):
        super(SegFormerHead, self).__init__(
            input_transform='multiple_select', **kwargs)

        decoder_params = kwargs['decoder_params']
        embedding_dim = decoder_params['embed_dim']
        conv_kernel_size = decoder_params['conv_kernel_size']

        self.linear_c = {}
        for i, in_channels in zip(self.in_index, self.in_channels):
            self.linear_c[str(i)] = MLP(
                input_dim=in_channels, embed_dim=embedding_dim)
        self.linear_c = nn.ModuleDict(self.linear_c)

        self.linear_fuse = ConvModule(
            in_channels=embedding_dim * len(self.in_index),
            out_channels=embedding_dim,
            kernel_size=conv_kernel_size,
            padding=0 if conv_kernel_size == 1 else conv_kernel_size // 2,
            norm_cfg=kwargs['norm_cfg'])

        self.linear_pred = nn.Conv2d(
            embedding_dim, self.num_classes, kernel_size=1)

    def forward(self, inputs):
        x = inputs
        n, _, h, w = x[-1].shape
        # for f in x:
        #     print(f.shape)

        _c = {}
        for i in self.in_index:
            # mmcv.print_log(f'{i}: {x[i].shape}, {self.linear_c[str(i)]}')
            _c[i] = self.linear_c[str(i)](x[i]).permute(0, 2, 1).contiguous()
            _c[i] = _c[i].reshape(n, -1, x[i].shape[2], x[i].shape[3])
            if i != 0:
                _c[i] = resize(
                    _c[i],
                    size=x[0].size()[2:],
                    mode='bilinear',
                    align_corners=False)

        _c = self.linear_fuse(torch.cat(list(_c.values()), dim=1))

        if self.dropout is not None:
            x = self.dropout(_c)
        else:
            x = _c
        x = self.linear_pred(x)

        return x