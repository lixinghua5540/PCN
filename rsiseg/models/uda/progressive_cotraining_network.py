# ---------------------------------------------------------------
# Copyright (c) 2026-2027 Jianhao MIao, Wuhan University. All rights reserved.
# Licensed under the Apache License, Version 2.0
# ---------------------------------------------------------------
# The ema model update and the domain-mixing are based on:
# https://github.com/vikolss/DACS
# Copyright (c) 2020 vikolss. Licensed under the MIT License.

import math
import random
from copy import deepcopy

import mmcv
from mmcv.parallel import DataContainer
from mmcv.runner import auto_fp16
import numpy as np
import torch
from timm.models.layers import DropPath
from torch.nn.modules.dropout import _DropoutNd
import torch.nn as nn
import torch.nn.functional as F

from rsiseg.core import add_prefix
from rsiseg.models import UDA, build_segmentor, builder
from rsiseg.models.backbones.vgg import Vgg19
from rsiseg.models.uda.uda_decorator import UDADecorator, get_module
from rsiseg.models.utils.dacs_transforms import ( get_class_masks,
                                                get_mean_std,weak_transform, strong_transform)
from rsiseg.ops import resize
from ..resunet import MultiModelEnsemble, NLayerDiscriminator, GANLoss, NewUnetGenerator
from ..ssim import MSSSIM
from rsiseg.models.utils.feat_center import *
import math
import pywt
from PIL import Image

def calc_grad_magnitude(grads, norm_type=2.0):
    norm_type = float(norm_type)
    if norm_type == math.inf:
        norm = max(p.abs().max() for p in grads)
    else:
        norm = torch.norm(
            torch.stack([torch.norm(p, norm_type) for p in grads]), norm_type)

    return norm

@UDA.register_module()
class Progressive_Cotraining_Network(UDADecorator):
    def __init__(self, **cfg):
        super(Progressive_Cotraining_Network, self).__init__(**cfg)
        self.local_iter = 0 #reset this parameter
        self.max_iters = cfg['max_iters']
        self.alpha = cfg['alpha']
        self.pseudo_threshold = cfg['pseudo_threshold']
        self.psweight_ignore_top = cfg['pseudo_weight_ignore_top']
        self.psweight_ignore_bottom = cfg['pseudo_weight_ignore_bottom']
        self.mix = cfg['mix']
        self.blur = cfg['blur']
        self.color_jitter_s = cfg['color_jitter_strength']
        self.color_jitter_p = cfg['color_jitter_probability']
        self.print_grad_magnitude = cfg['print_grad_magnitude']
        self.trg_loss_weight = cfg.get('trg_loss_weight', 1.)
        self.use_decoded_feats = cfg.get('use_decoded_feats', False)
        self.thre_type = cfg.get('thre_type', 'all')
        self.strong_aug_denorm_type = cfg.get('strong_aug_denorm_type', 'mean_std')
        self.apply_no_mix = cfg.get('apply_no_mix', False)
        self.with_semantic=True
        self.apply_ensemble=False#
        self.num_classes=19
        self.patch_size=256
        self.Ds = NLayerDiscriminator(3, 4, 1, ndf=64, n_layers=4, norm_layer="InstanceNorm2d")
        self.Dt = NLayerDiscriminator(3, 4, 1, ndf=64, n_layers=4, norm_layer="InstanceNorm2d")
        self.Gs = NewUnetGenerator(input_nc=3, output_nc=3, ngf=64, num_classes=self.num_classes, patch_size=self.patch_size, with_semantic=self.with_semantic, norm_layer="InstanceNorm2d")
        self.Gt = NewUnetGenerator(input_nc=3, output_nc=3, ngf=64, num_classes=self.num_classes, patch_size=self.patch_size, with_semantic=self.with_semantic, norm_layer="InstanceNorm2d")
        self.ensemble_weight= MultiModelEnsemble(n_models=2, n_classes=self.num_classes)
        self.ssim_loss=MSSSIM(window_size=11, size_average=True, channel=3)
        self.G_semantic_loss=nn.CrossEntropyLoss(ignore_index=255)
        self.criterionGAN = GANLoss("lsgan")  # define GAN loss.
        self.criterionCycle = torch.nn.L1Loss()
        self.criterionRIL = torch.nn.L1Loss()
        self.criterionContent = torch.nn.L1Loss()
        self.vgg_pretrained = Vgg19(vgg19_npy_path="./pretrained/vgg19.npy")
        self.vgg_pretrained.load_dict()
        self.vgg_pretrained.eval()
        self.img_dir="./work_dirs/image"
        self.taskname="progressive_cotraining_network"

        assert self.mix == 'class'
        self.entropy_weight = 1.0
        ema_cfg = deepcopy(cfg['model'])
        dual_cfg = deepcopy(cfg['model'])
        dual_cfg["auxiliary_head"]=None

        self.ema_model = build_segmentor(ema_cfg)
        self.dual_model = build_segmentor(dual_cfg)

        aux_losses = cfg.get('aux_losses', None)
        self.apply_aux = False
        if aux_losses is not None:
            self.apply_aux = True
            if not type(aux_losses) == list:
                aux_losses = [aux_losses]

            aux_losses = [builder.build_loss(loss) for loss in aux_losses]
            self.aux_losses = nn.ModuleList(aux_losses)

    def get_ema_model(self):
        return get_module(self.ema_model)
    
    def get_dual_model(self):
        return get_module(self.dual_model)

    def _init_ema_weights(self):
        for param in self.get_ema_model().parameters():
            param.detach_()
        mp = list(self.get_model().parameters())
        mcp = list(self.get_ema_model().parameters())
        for i in range(0, len(mp)):
            if not mcp[i].data.shape:  # scalar tensor
                mcp[i].data = mp[i].data.clone()
            else:
                mcp[i].data[:] = mp[i].data[:].clone()

    def _update_ema(self, iter):
        alpha_teacher = min(1 - 1 / (iter + 1), self.alpha)
        for ema_param, param in zip(self.get_ema_model().parameters(),
                                    self.get_model().parameters()):
            if not param.data.shape:  # scalar tensor
                ema_param.data = \
                    alpha_teacher * ema_param.data + \
                    (1 - alpha_teacher) * param.data
            else:
                ema_param.data[:] = \
                    alpha_teacher * ema_param[:].data[:] + \
                    (1 - alpha_teacher) * param[:].data[:]

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        log_vars, vis_states = self(**data_batch, **optimizer)
        log_vars.pop('loss', None)
        outputs = dict(
            log_vars=log_vars,
            num_samples=len(data_batch['img_metas']),
            states=vis_states
        )
        return outputs
    
    @auto_fp16() 
    def forward_train(self, img, img_metas, img_copy, gt_semantic_seg, target_img,
                      target_img_metas, target_img_copy, target_img_strong_aug, optim1, optim2, optim3):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `rsiseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        log_vars = {}
        vis_states = {}
        total_loss = 0
        total_loss2 = 0
        batch_size = img.shape[0]
        dev = img.device

        # Init/update ema model
        if self.local_iter == 0:
            self._init_ema_weights()

        if self.local_iter > 0:
            self._update_ema(self.local_iter)

        means, stds = get_mean_std(img_metas, dev)
        strong_parameters = {
            'mix': None,
            'color_jitter': random.uniform(0, 1),
            'color_jitter_s': self.color_jitter_s,
            'color_jitter_p': self.color_jitter_p,
            'blur': random.uniform(0, 1) if self.blur else 0,
            'mean': means[0].unsqueeze(0),  # assume same normalization
            'std': stds[0].unsqueeze(0),
            'denorm_type': self.strong_aug_denorm_type
        }

        optim2.zero_grad()
        # Generate pseudo-label
        for m in self.get_ema_model().modules():
            if isinstance(m, _DropoutNd):
                m.training = False
            if isinstance(m, DropPath):
                m.training = False

        ema_logits, ema_states = self.get_ema_model().encode_decode(
            target_img, target_img_metas)
        ema_feats = ema_states['feats']
        if self.use_decoded_feats:
            ema_feats = ema_states['decoded_features']
        ema_softmax = torch.softmax(ema_logits.detach(), dim=1)
        pseudo_prob, pseudo_label = torch.max(ema_softmax, dim=1)
        ps_large_p = pseudo_prob.ge(self.pseudo_threshold).long() == 1
        ps_size = np.size(np.array(pseudo_label.cpu()))
        translation_logits, _ = self.get_dual_model().encode_decode(#new pseudo label
            target_img_copy, target_img_metas)#fakes
        translation_softmax = torch.softmax(translation_logits.detach(), dim=1)
        #translation pseudo label

        #combine the pseudo information from ema teacher model and translation segmentation model
        comb_logits = (ema_logits+translation_logits)/2 
        comb_softmax = torch.softmax(comb_logits.detach(), dim=1)
        comb_pseudo_prob, comb_pseudo_label=torch.max(comb_softmax, dim=1)
        comb_ps_large_p = comb_pseudo_prob.ge(self.pseudo_threshold).long() == 1
        comb_ps_size = np.size(np.array(comb_pseudo_label.cpu()))
        
        # JS divergence
        js_dist = self.js_divergence(ema_softmax, translation_softmax)
        distribution_similarity = torch.exp(-js_dist)  # 转换为相似度分数
        # the consistency of dual prediction
        continuous_agreement = torch.sum(ema_softmax * translation_softmax, dim=1)
        pseudo_consistency = (distribution_similarity+continuous_agreement)/2.0+0.3
        # entropy-based pseudo filter, class entropy and spatial entropy
        entropy_ema = -torch.sum(ema_softmax.detach() * torch.log(ema_softmax.detach() + 1e-10), dim=1)  # [B, H, W]
        max_entropy_ema = torch.log(torch.tensor(ema_softmax.size(1), device=dev))
        basic_entropy_ema = entropy_ema / max_entropy_ema  # normalize to [0,1]
        nse_ema = self.optimized_structural_entropy(ema_softmax.detach())
        nge_ema = self.optimized_vectorized_graph_entropy(ema_softmax.detach())
        entropy_weight_ema = torch.clamp((2.5 - basic_entropy_ema-0.5*nse_ema-nge_ema), min=0)
        B = entropy_weight_ema.shape[0]
        min_vals_ema = entropy_weight_ema.view(B, -1).min(dim=1)[0].view(B, 1, 1)
        max_vals_ema = entropy_weight_ema.view(B, -1).max(dim=1)[0].view(B, 1, 1)
        range_vals_ema = max_vals_ema - min_vals_ema
        range_vals_ema[range_vals_ema == 0] = 1 
        normalized_entropy_ema = (entropy_weight_ema - min_vals_ema) / range_vals_ema
        
        entropy_weight_ema2 = 0.6 + self.local_iter/200000 + (0.8-self.local_iter/80000)*normalized_entropy_ema
        entropy = -torch.sum(comb_softmax * torch.log(comb_softmax + 1e-10), dim=1)  # [B, H, W]
        max_entropy = torch.log(torch.tensor(comb_softmax.size(1), device=dev))
        basic_entropy = entropy / max_entropy  # normalize to [0,1]
        nse = self.optimized_structural_entropy(comb_softmax)
        nge = self.optimized_vectorized_graph_entropy(comb_softmax)
        entropy_weight2 = torch.clamp((2.5 - basic_entropy-0.5*nse-nge), min=0)
        min_vals = entropy_weight2.view(B, -1).min(dim=1)[0].view(B, 1, 1)
        max_vals = entropy_weight2.view(B, -1).max(dim=1)[0].view(B, 1, 1)
        range_vals = max_vals - min_vals
        range_vals[range_vals == 0] = 1 
        normalized_entropy_comb = (entropy_weight2 - min_vals) / range_vals
        entropy_weight_comb = 0.6 +self.local_iter/200000+ (0.8-self.local_iter/80000)*normalized_entropy_comb
        #s_ratio = 0.4 + 0.3 * self.local_iter/40000
        s_ratio = 0.7
        entropy_weight_m = self.create_per_class_entropy_selection_mask(pseudo_consistency*entropy_weight2, labels = comb_logits.detach().argmax(axis=1).unsqueeze(axis=1), selection_ratio=s_ratio)
        # image translation stage
        if self.with_semantic==True:
            fake_s, sflabel, s_mid = self.Gs(img_copy)
            fake_t, _, t_mid  = self.Gt(target_img_copy)
            rec_s, tflabel, s_midf  = self.Gt(fake_s)
            rec_t, _, t_midf   = self.Gs(fake_t)
        else:
            fake_s = self.Gs(img_copy)
            fake_t = self.Gt(target_img_copy)
            rec_s = self.Gt(fake_s)
            rec_t = self.Gs(fake_t)

        loss_ssim=2-self.ssim_loss(fake_t, target_img_copy)-self.ssim_loss(fake_s, img_copy)

        loss_cycle=self.criterionCycle(img_copy, rec_s)+self.criterionCycle(target_img_copy, rec_t)#or it can use half of the loss
        nambda_ssim = 2 * (1-self.local_iter/80000)#
        #nambda_ssim = 2.0 #
        loss_generative = loss_cycle+nambda_ssim*loss_ssim

        if self.local_iter<40000:
            sfake_loss = self.criterionGAN(self.Ds(fake_t),True)
            tfake_loss = self.criterionGAN(self.Dt(fake_s),True)
            loss_D=sfake_loss+tfake_loss
            loss_generative += loss_D

        # add a represent invariant loss, has not be utlized in this version
        if self.with_semantic==True:
            g_label_loss=self.G_semantic_loss(sflabel, gt_semantic_seg.squeeze().long())
            t_label_loss=self.G_semantic_loss(tflabel, gt_semantic_seg.squeeze().long())
            nambda=2*math.exp(-2*(1-self.local_iter/40000)**2)#2 -2
            total_loss += (loss_generative + nambda*(g_label_loss + t_label_loss))
        else:
            total_loss += loss_generative

        if self.local_iter>=4000:#4000
            with torch.no_grad():
                feat_img_copy=self.vgg_pretrained(img_copy)
                align_class_center_source = class_center_cal1(feat_img_copy, gt_semantic_seg, num_classes=self.num_classes)
            feat_fake_t=self.vgg_pretrained(fake_t)
            class_center_fake_t = class_center_cal_with_external_selection_org(feat_fake_t, comb_logits.detach().argmax(axis=1).unsqueeze(axis=1), selection_mask=entropy_weight_m, num_classes=self.num_classes)
            diff_weight = [0.0002, 0.000018, 0.0000036]
            if isinstance(class_center_fake_t, list):
                center_align_loss1 = 0
                i = 0
                for cs_fake_t, align_src in zip(class_center_fake_t, align_class_center_source):
                    # to add a threshold to constrain center alignment
                    if i<3:
                        diff = cs_fake_t - align_src.detach()
                        center_align_loss1 = center_align_loss1 + diff_weight[i] * torch.pow(diff, 2).sum()
                        i = i + 1
            else:
                center_diff1 = class_center_fake_t - align_class_center_source.detach()
                center_align_loss1 = 0.1 * torch.pow(center_diff1, 2).sum()
            with torch.no_grad():
                feat_target_img=self.vgg_pretrained(target_img_copy)
                align_class_center_target = class_center_cal_with_external_selection_org(feat_target_img, comb_logits.detach().argmax(axis=1).unsqueeze(axis=1), selection_mask=entropy_weight_m, num_classes=self.num_classes)
            feat_fake_s=self.vgg_pretrained(fake_s)
            class_center_fake_s = class_center_cal1(feat_fake_s, gt_semantic_seg, num_classes=self.num_classes) # t like  
          # Handle tensor list inputs
            if isinstance(class_center_fake_s, list):
                center_align_loss2 = 0
                i = 0
                for cs_fake_s, align_tar in zip(class_center_fake_s, align_class_center_target):
                    if i<3:
                        diff = cs_fake_s - align_tar.detach()
                        center_align_loss2 = center_align_loss2 +  diff_weight[i] * torch.pow(diff, 2).sum()
                        i = i + 1
            else:
                center_diff2 = class_center_fake_s - align_class_center_target.detach()
                center_align_loss2 = 0.1 * torch.pow(center_diff2, 2).sum()
            total_loss += (center_align_loss1+center_align_loss2)

        if self.local_iter<=40000:
            total_loss.backward()
            optim2.step()

        if self.local_iter<40000:#
            optim3.zero_grad()
            loss_adversarial = self.criterionGAN(self.Dt(fake_s.detach()),False)+self.criterionGAN(self.Ds(fake_t.detach()),False)+self.criterionGAN(self.Dt(target_img_copy),True)+self.criterionGAN(self.Ds(img_copy),True)
            loss_adversarial.backward()
            optim3.step()

        # training segmentation
        optim1.zero_grad()
        # Train on source images
        clean_losses = self.get_model().forward_train(
            img, img_metas, gt_semantic_seg, return_feats=True, return_logits=True,
            return_decoded_feats=self.use_decoded_feats
        )
        src_feats = clean_losses.pop('features')
        if self.use_decoded_feats:
            src_feats = clean_losses.pop('decoded_features')
        src_logits = clean_losses.pop('logits')
        clean_loss, clean_log_vars = self._parse_losses(clean_losses)
        log_vars.update(clean_log_vars)

        total_loss2 += clean_loss

        if self.print_grad_magnitude:
            params = self.get_model().backbone.parameters()
            seg_grads = [
                p.grad.detach().clone() for p in params if p.grad is not None
            ]
            grad_mag = calc_grad_magnitude(seg_grads)
            mmcv.print_log(f'Seg. Grad.: {grad_mag}', 'rsiseg')

        if self.thre_type == 'all':
            pseudo_weight = torch.sum(ps_large_p).item() / ps_size
            pseudo_weight = pseudo_weight * torch.ones(pseudo_prob.shape, device=dev)
            comb_pseudo_weight = torch.sum(comb_ps_large_p).item() / comb_ps_size
            comb_pseudo_weight = comb_pseudo_weight * torch.ones(comb_pseudo_prob.shape, device=dev)
        elif self.thre_type == 'part':
            pseudo_weight = ps_large_p.float()
            comb_pseudo_weight = comb_ps_large_p.float()

        # Don't trust pseudo-labels in regions with potential
        # rectification artifacts. This can lead to a pseudo-label
        # drift from sky towards building or traffic light.
        if self.psweight_ignore_top > 0:
            pseudo_weight[:, :self.psweight_ignore_top, :] = 0
        if self.psweight_ignore_bottom > 0:
            pseudo_weight[:, -self.psweight_ignore_bottom:, :] = 0
        gt_pixel_weight = torch.ones((pseudo_weight.shape), device=dev)
        gt_pixel_entropy = torch.ones((pseudo_weight.shape), device=dev)
        gt_pixel_entropy_ema = torch.ones((pseudo_weight.shape), device=dev)
        # Apply mixing
        mixed_img, mixed_lbl = [None] * batch_size, [None] * batch_size
        mix_masks = get_class_masks(gt_semantic_seg)
        if self.apply_no_mix:
            for mix_mask in mix_masks:
                mix_mask[:] = 0

        for i in range(batch_size):
            strong_parameters['mix'] = mix_masks[i]
            trg_img =  target_img[i] if self.apply_no_mix else target_img_strong_aug[i]
            mixed_img[i], mixed_lbl[i] = strong_transform(
                strong_parameters,
                data=torch.stack((img[i], trg_img)),
                target=torch.stack((gt_semantic_seg[i][0], pseudo_label[i]))
            )
            _, pseudo_weight[i] = strong_transform(
                strong_parameters,
                target=torch.stack((gt_pixel_weight[i], pseudo_weight[i]))
            )

        mixed_img = torch.cat(mixed_img)
        mixed_lbl = torch.cat(mixed_lbl)

        #Enhance translation image by Wavelet transform
        with torch.no_grad():
            Ar, (Hr, Vr, Dr) = pywt.dwt2(img_copy.cpu().detach(),'haar')
            Af, (Hf, Vf, Df) = pywt.dwt2(fake_s.cpu().detach(),'haar')
            fake_s_dwt = torch.tensor(pywt.idwt2((Af,(Hr, Vr, Dr)),'haar')).to(dev)
        #fake_s_dwt = torch.tensor(pywt.idwt2((Af,(Hf, Vf, Dr)),'haar')).to(dev)
        # mixing trans image
        mixed_img_t, mixed_label_t = [None] * batch_size, [None] * batch_size
        mixed_pseudo_consistency =  [None] * batch_size
        entropy_weight_mix =  [None] * batch_size
        entropy_weight_mix_ema =  [None] * batch_size
        mixed_parameter=deepcopy(strong_parameters)#no augmentation
        for i in range(batch_size):
            mixed_parameter['mix'] = mix_masks[i]
            trg_img_t = deepcopy(target_img_copy[i])
            mixed_img_t[i], mixed_label_t[i] = weak_transform(
                mixed_parameter,
                data=torch.stack((fake_s_dwt[i].detach(), trg_img_t.detach())),
                target=torch.stack((gt_semantic_seg[i][0],comb_pseudo_label[i]))
            )
            _, comb_pseudo_weight[i] = weak_transform(
                mixed_parameter,#
                target=torch.stack((gt_pixel_weight[i], comb_pseudo_weight[i]))
            )
            _, entropy_weight_mix[i] = weak_transform(
                mixed_parameter,#
                target=torch.stack((gt_pixel_entropy[i], entropy_weight_comb[i]))
            )
            _, entropy_weight_mix_ema[i] = weak_transform(
                mixed_parameter,
                target=torch.stack((gt_pixel_entropy_ema[i], entropy_weight_ema2[i]))
            )
            _, mixed_pseudo_consistency[i] = weak_transform(
                mixed_parameter,
                target=torch.stack((gt_pixel_weight[i], pseudo_consistency[i]))
            )
        mixed_img_t = torch.cat(mixed_img_t)
        mixed_label_t = torch.cat(mixed_label_t)
        mixed_pseudo_consistency = torch.cat(mixed_pseudo_consistency).squeeze()#
        entropy_weight_mix = torch.cat(entropy_weight_mix).squeeze()
        entropy_weight_mix_ema = torch.cat(entropy_weight_mix_ema).squeeze()

        mix_losses = self.get_model().forward_train(
            mixed_img, img_metas, mixed_lbl, pseudo_weight * mixed_pseudo_consistency, return_feats=True, return_logits=True)#mixed_pseudo_consistency
        mix_losses_t = self.get_dual_model().forward_train(
            mixed_img_t, img_metas, mixed_label_t, comb_pseudo_weight * entropy_weight_mix * mixed_pseudo_consistency, return_feats=False, return_logits=True) # *entropy_weight_mix
        mix_losses_t_org = self.get_dual_model().forward_train( # target image loss calculation
            fake_s_dwt.detach(), img_metas, gt_semantic_seg, return_feats=False, return_logits=True) #fakes this is gt semantic seg so the logits can be true
        
        mixed_feats = mix_losses.pop('features') 
        mixed_logits = mix_losses.pop('logits')
        mixed_logits_t = mix_losses_t.pop('logits')
        tar_logits = mix_losses_t_org.pop('logits')
        mix_losses = add_prefix(mix_losses, 'mix')
        mix_loss, mix_log_vars = self._parse_losses(mix_losses)
        log_vars.update(mix_log_vars)
        total_loss2 += mix_loss * self.trg_loss_weight
        mix_losses_t = add_prefix(mix_losses_t, 'mix_t')#
        mix_losses_t, mix_log_vars_t = self._parse_losses(mix_losses_t)#
        log_vars.update(mix_log_vars_t)#
        mix_losses_t_org = add_prefix(mix_losses_t_org, 'mix_t_org')#
        mix_losses_t_org, mix_log_vars_t_org = self._parse_losses(mix_losses_t_org)#
        log_vars.update(mix_log_vars_t_org)#
        total_loss2 += 2*(mix_losses_t+mix_losses_t_org)
        
        tensors = dict(
            img_src=img,
            img_src_metas=img_metas,
            # img_trg=target_img,
            img_trg=mixed_img,
            img_mixed=mixed_img,
            img_metas_trg=target_img_metas,
            gt_src=gt_semantic_seg,
            x_src=src_feats,
            x_ema=ema_feats,
            # x_ema = mixed_ema_feats,
            # x_trg = trg_feats,
            x_trg = mixed_feats,
            logits_src=src_logits,
            logits_trg=mixed_logits,
            logits_ema=ema_logits,
            mix_masks=torch.cat(mix_masks, dim=0),
            pseudo_weight=pseudo_weight
            # logits_trg=trg_logits
        )

        if self.apply_aux:
            aux_losses = self._get_aux_losses(tensors=tensors)#aux losses calculate the 
            vis_states.update({name: value for name, value in aux_losses.items() if name.startswith('vis|')})
            for name in vis_states.keys(): aux_losses.pop(name)

            aux_losses, aux_log_vars = self._parse_losses(aux_losses)
            log_vars.update(aux_log_vars)
            # aux_losses.backward()
            total_loss2 += aux_losses

        total_loss2.backward()
        optim1.step()

        vis_mask_mix = mixed_lbl
        vis_pseudo_weight = F.interpolate(pseudo_weight.unsqueeze(1), mixed_lbl.shape[2:])
        vis_mask_mix = torch.where(vis_pseudo_weight > 0.0, vis_mask_mix, 255)
        vis_states.update({
            'vis|seg_mask_src': (img, gt_semantic_seg, src_logits.max(dim=1)[1].unsqueeze(1)),
            'vis|seg_mask_mix': (mixed_img, vis_mask_mix, mixed_logits.max(dim=1)[1].unsqueeze(1).float()),
        })
        #output the image translation result
        if self.local_iter%1000==0:
            #denorm the result and output by PIL
            for i in range(fake_t.shape[0]):
                batch_t=fake_t[i]
                batch_s=fake_s[i]
                dwt_s=fake_s_dwt[i]
                NormBatchdata_t=batch_t.permute([1,2,0])*torch.tensor([58.395, 57.12, 57.375],device="cuda")+torch.tensor([123.675, 116.28, 103.53],device="cuda")#*torch.tensor([58.395, 57.12, 57.375])
                NormBatchdata_s=batch_s.permute([1,2,0])*torch.tensor([58.395, 57.12, 57.375],device="cuda")+torch.tensor([123.675, 116.28, 103.53],device="cuda")
                pil_t=Image.fromarray(NormBatchdata_t.byte().cpu().numpy())
                pil_s=Image.fromarray(NormBatchdata_s.byte().cpu().numpy())
                pil_t.save(self.img_dir+"/"+self.taskname+"/fake_t_iter_"+str(self.local_iter)+"_"+str(i)+".png")
                pil_s.save(self.img_dir+"/"+self.taskname+"/fake_s_iter_"+str(self.local_iter)+"_"+str(i)+".png")
                
                NormBatchdata_s_dwt=dwt_s.permute([1,2,0])*torch.tensor([58.395, 57.12, 57.375],device="cuda")+torch.tensor([123.675, 116.28, 103.53],device="cuda")
                pil_dwt=Image.fromarray(NormBatchdata_s_dwt.byte().cpu().numpy())
                pil_dwt.save(self.img_dir+"/"+self.taskname+"/fake_s_dwt_iter_"+str(self.local_iter)+"_"+str(i)+".png")
                NormBatchdata_rt=target_img_copy[i].permute([1,2,0])*torch.tensor([58.395, 57.12, 57.375],device="cuda")+torch.tensor([123.675, 116.28, 103.53],device="cuda")#*torch.tensor([58.395, 57.12, 57.375])
                NormBatchdata_rs=img_copy[i].permute([1,2,0])*torch.tensor([58.395, 57.12, 57.375],device="cuda")+torch.tensor([123.675, 116.28, 103.53],device="cuda")
                pil_rt=Image.fromarray(NormBatchdata_rt.byte().cpu().numpy())
                pil_rs=Image.fromarray(NormBatchdata_rs.byte().cpu().numpy())
                pil_rt.save(self.img_dir+"/"+self.taskname+"/real_t_iter_"+str(self.local_iter)+"_"+str(i)+".png")
                pil_rs.save(self.img_dir+"/"+self.taskname+"/real_s_iter_"+str(self.local_iter)+"_"+str(i)+".png")
                #mix img
                NormBatchdata_mix=mixed_img[i].permute([1,2,0])*torch.tensor([58.395, 57.12, 57.375],device="cuda")+torch.tensor([123.675, 116.28, 103.53],device="cuda")#*torch.tensor([58.395, 57.12, 57.375])
                pil_mix=Image.fromarray(NormBatchdata_mix.byte().cpu().numpy())
                pil_mix.save(self.img_dir+"/"+self.taskname+"/mix_iter_"+str(self.local_iter)+"_"+str(i)+".png")
                NormBatchdata_mix_t=mixed_img_t[i].permute([1,2,0])*torch.tensor([58.395, 57.12, 57.375],device="cuda")+torch.tensor([123.675, 116.28, 103.53],device="cuda")#*torch.tensor([58.395, 57.12, 57.375])
                pil_mix_t=Image.fromarray(NormBatchdata_mix_t.byte().cpu().numpy())
                pil_mix_t.save(self.img_dir+"/"+self.taskname+"/mix_t_iter_"+str(self.local_iter)+"_"+str(i)+".png")

        self.local_iter += 1

        return log_vars, vis_states
    
    def forward_test(self, imgs, img_metas,**kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            if isinstance(img_meta, DataContainer):
                img_meta = img_meta.data
                img_meta = img_meta[0]
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            feature_level_pred = self.simple_test_feature(imgs[0], img_metas[0], **kwargs)
            img_level_pred = self.simple_test_feature_dual(imgs[0], img_metas[0], **kwargs)
        else:
            feature_level_pred = self.aug_test(imgs, img_metas, **kwargs)
            img_level_pred = self.simple_test_feature_dual(imgs, img_metas, **kwargs)
        if self.apply_ensemble==False:
            for i in range(len(feature_level_pred[0])):
                test_final=feature_level_pred[0][i]+img_level_pred[0][i]
                feature_level_array=feature_level_pred[0][i].argmax(axis=0)
                feature_level_pred[0][i]=feature_level_array
                img_level_array=img_level_pred[0][i].argmax(axis=0)
                img_level_pred[0][i]=img_level_array

        else:
            test_final = self.ensemble_weight(torch.stack([torch.from_numpy(feature_level_pred[0][0]).unsqueeze(0), torch.from_numpy(img_level_pred[0][0]).unsqueeze(0)]).to("cuda")).squeeze().cpu().numpy()
            for i in range(len(feature_level_pred[0])):
                feature_level_array=feature_level_pred[0][i].argmax(axis=0)
                feature_level_pred[0][i]=feature_level_array
                img_level_array=img_level_pred[0][i].argmax(axis=0)
                img_level_pred[0][i]=img_level_array
        return feature_level_pred, img_level_pred, test_final


    def forward_validation(self, imgs, img_metas,**kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.
        """
        for var, name in [(imgs, 'imgs'), (img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got '
                                f'{type(var)}')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) != '
                             f'num of image meta ({len(img_metas)})')
        # all images in the same aug batch all of the same ori_shape and pad
        # shape
        for img_meta in img_metas:
            if isinstance(img_meta, DataContainer):
                img_meta = img_meta.data
                img_meta = img_meta[0]
            ori_shapes = [_['ori_shape'] for _ in img_meta]
            assert all(shape == ori_shapes[0] for shape in ori_shapes)
            img_shapes = [_['img_shape'] for _ in img_meta]
            assert all(shape == img_shapes[0] for shape in img_shapes)
            pad_shapes = [_['pad_shape'] for _ in img_meta]
            assert all(shape == pad_shapes[0] for shape in pad_shapes)

        if num_augs == 1:
            trans_img = self.translation(imgs[0], img_metas[0], **kwargs)
            feature_level_pred = self.simple_test_feature(trans_img, img_metas[0], **kwargs)
            img_level_pred = self.simple_test_feature_dual(trans_img, img_metas[0], **kwargs)# it is believe this is an image level alignment
        else:
            trans_img = self.translation(imgs, img_metas, **kwargs)
            feature_level_pred = self.aug_test(trans_img, img_metas, **kwargs)
            img_level_pred = self.simple_test_feature_dual(trans_img, img_metas, **kwargs)
        if self.apply_ensemble==False:
            for i in range(len(feature_level_pred[0])):
                test_final=feature_level_pred[0][i]+img_level_pred[0][i]# is believed to be a combination of feature level and image level segmentaiton
                feature_level_array=feature_level_pred[0][i].argmax(axis=0)
                feature_level_pred[0][i]=feature_level_array
                img_level_array=img_level_pred[0][i].argmax(axis=0)
                img_level_pred[0][i]=img_level_array

        else:
            test_final = self.ensemble_weight(torch.stack([torch.from_numpy(feature_level_pred[0][0]).unsqueeze(0), torch.from_numpy(img_level_pred[0][0]).unsqueeze(0)]).to("cuda")).squeeze().cpu().numpy()
            for i in range(len(feature_level_pred[0])):
                feature_level_array=feature_level_pred[0][i].argmax(axis=0)
                feature_level_pred[0][i]=feature_level_array
                img_level_array=img_level_pred[0][i].argmax(axis=0)
                img_level_pred[0][i]=img_level_array
        return feature_level_pred, img_level_pred, test_final

    @auto_fp16(apply_to=('img', ))
    def forward(self, img, img_metas, return_loss=True, validation=False, **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(img, img_metas, **kwargs)
        else:
            if validation:
                return self.forward_validation(img, img_metas, **kwargs)
            else:
                return self.forward_test(img, img_metas, **kwargs)

    def translation(self, img, img_meta, rescale=True, **kwargs):
        """Inference semantic branch of target domain by sliding-window with overlap.

        If h_crop > h_img or w_crop > w_img, the small patch will be used to
        decode without padding.
        """
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, channel, h_img, w_img = img.size()
        num_classes = self.num_classes
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, channel, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                if self.with_semantic == True:
                    trans_crop_img, _ = self.Gs(crop_img)#from the source domain to the target domain
                else:
                    trans_crop_img = self.Gs(crop_img)#from the source domain to the target domain
                #image translation
                preds += F.pad(trans_crop_img,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))

                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        if rescale:
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=True,
                warning=False)
        return preds
    
    def optimized_structural_entropy(self, probs, window_size=5):
        B, C, H, W = probs.shape
        padding = window_size // 2
        # the sum of probability of every class in the neighbor
        kernel = torch.ones(1, 1, window_size, window_size, device=probs.device)
        expanded_kernel = kernel.repeat(C, 1, 1, 1)
        neighbor_sum = F.conv2d(probs, expanded_kernel, padding=padding, groups=C)
        neighbor_sum_norm = neighbor_sum / (neighbor_sum.sum(dim=1, keepdim=True) + 1e-12)
        #calculate the entropy
        log_probs = torch.log(torch.clamp(neighbor_sum_norm, min=1e-12, max=1.0))
        entropy = -torch.sum(neighbor_sum_norm * log_probs, dim=1)
        max_entropy = math.log(min(window_size * window_size, C))
        #normalize the entropy
        normalized_entropy = entropy / (max_entropy + 1e-12)
        
        return torch.clamp(normalized_entropy, 0.0, 1.0)

    def optimized_vectorized_graph_entropy(self, predictions, connectivity=8):
        B, C, H, W = predictions.shape
        hard_labels = torch.argmax(predictions, dim=1).long()
        if connectivity == 8:
            kernel = torch.ones(1, 1, 3, 3, device=predictions.device)
            neighbor_count = 9
        else:
            kernel = torch.zeros(1, 1, 3, 3, device=predictions.device)
            kernel[0, 0, 0, 1] = 1
            kernel[0, 0, 1, 0] = 1
            kernel[0, 0, 1, 2] = 1
            kernel[0, 0, 2, 1] = 1
            kernel[0, 0, 1, 1] = 1
            neighbor_count = 5
        one_hot = F.one_hot(hard_labels, num_classes=C).permute(0, 3, 1, 2).float()  # [B, C, H, W]
        # Compute valid neighbor mask (accounting for image boundaries)
        valid_neighbor_mask = F.conv2d(
            torch.ones(B, 1, H, W, device=predictions.device),
            kernel, padding=1
        )
        # Compute neighborhood distribution for each class (including center pixel)
        expanded_kernel = kernel.repeat(C, 1, 1, 1)  # [C, 1, 3, 3]
        neighbor_distributions = F.conv2d(
            one_hot, expanded_kernel, padding=1, groups=C
        )
        # Compute probability distribution in neighborhood
        neighbor_probs = neighbor_distributions / (valid_neighbor_mask + 1e-12)  # [B, C, H, W]
        # Compute consistency measure: probability of center pixel's class in its neighborhood
        center_class_idx = hard_labels.unsqueeze(1)  # [B, 1, H, W]
        # Gather the probability of center pixel's class in the neighborhood distribution
        center_class_neighbor_prob = torch.gather(neighbor_probs, 1, center_class_idx).squeeze(1)  # [B, H, W]
        # Compute consistency entropy: use negative log to measure inconsistency, Low entropy when center class probability is high (good consistency), High entropy when center class probability is low (poor consistency)
        consistency_entropy = -torch.log(center_class_neighbor_prob + 1e-12)
        max_entropy = -torch.log(torch.tensor(1.0 / neighbor_count, device=predictions.device) + 1e-12)
        normalized_entropy = consistency_entropy / (max_entropy + 1e-12)
        return torch.clamp(normalized_entropy, 0.0, 1.0)
    
    def create_per_class_entropy_selection_mask(self, entropy_weight_all, labels, selection_ratio=0.3):
        """Create selection mask based on entropy values for each class separately
        
        Args:
            entropy_weight_all: entropy matrix [N, H, W] indicating uncertainty (higher = more confident)
            labels: pseudo labels [N, H, W] 
            selection_ratio: ratio of lowest-entropy pixels to select per class
            
        Returns:
            Binary mask [N, H, W] where 1 indicates selected pixels
        """
        n, h, w = entropy_weight_all.shape
        device = entropy_weight_all.device
        # Initialize selection mask with zeros
        selection_mask = torch.zeros(n, h, w, device=device, dtype=torch.float32)
        for i in range(n):
            # Get current sample's entropy and labels
            sample_entropy = entropy_weight_all[i]  # [H, W]
            sample_labels = labels[i]  # [H, W]
            # Flatten for easier processing
            entropy_flat = sample_entropy.view(-1)  # [H*W]
            labels_flat = sample_labels.view(-1)  # [H*W]
            # Get unique classes in this sample (excluding ignore label if present)
            unique_classes = torch.unique(sample_labels)
            unique_classes = unique_classes[unique_classes != 255]  # Remove ignore label if exists
            for cls in unique_classes:
                # Get indices of pixels belonging to this class
                cls_mask = (labels_flat == cls)
                cls_indices = torch.where(cls_mask)[0]
                if len(cls_indices) == 0:
                    continue
                # Get entropy values for this class
                cls_entropy = entropy_flat[cls_indices]
                # Calculate how many pixels to select for this class
                select_count = max(1, int(len(cls_indices) * selection_ratio))
                # Select top-k most confident pixels (highest entropy_weight_all values)
                _, topk_indices = torch.topk(cls_entropy, k=select_count, largest=True)
                # Map back to original flat indices
                selected_flat_indices = cls_indices[topk_indices]
                # Convert flat indices back to 2D coordinates
                selected_h = selected_flat_indices // w
                selected_w = selected_flat_indices % w
                # Mark these pixels in the selection mask
                selection_mask[i, selected_h, selected_w] = 1
        
        return selection_mask
    def js_divergence(self, p, q):
        eps = 1e-8
        p = torch.clamp(p, min=eps, max=1.0-eps)
        q = torch.clamp(q, min=eps, max=1.0-eps)
        m = 0.5 * (p + q)
        kl_pm = F.kl_div(torch.log(p), m, reduction='none').sum(dim=1)
        kl_qm = F.kl_div(torch.log(q), m, reduction='none').sum(dim=1)
        jsd = 0.5 * (kl_pm + kl_qm)
        return jsd
    
    def _get_aux_losses(self, tensors):

        aux_losses = dict()
        for loss_module in self.aux_losses:
            loss_ = loss_module(tensors)
            if loss_ is None:
                continue
            aux_losses.update(loss_)

        return aux_losses
