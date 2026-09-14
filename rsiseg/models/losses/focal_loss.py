# Copyright (c) OpenMMLab. All rights reserved.
# Modified from https://github.com/open-mmlab/mmdetection
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.ops import sigmoid_focal_loss as _sigmoid_focal_loss

from ..builder import LOSSES
from .utils import weight_reduce_loss

import numpy as np

# This method is used when cuda is not available
def py_sigmoid_focal_loss(pred,
                          target,
                          one_hot_target=None,
                          weight=None,
                          gamma=2.0,
                          alpha=0.5,
                          class_weight=None,
                          valid_mask=None,
                          reduction='mean',
                          avg_factor=None):
    """PyTorch version of `Focal Loss <https://arxiv.org/abs/1708.02002>`_.

    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the
            number of classes
        target (torch.Tensor): The learning label of the prediction with
            shape (N, C)
        one_hot_target (None): Placeholder. It should be None.
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float | list[float], optional): A balanced form for Focal Loss.
            Defaults to 0.5.
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        valid_mask (torch.Tensor, optional): A mask uses 1 to mark the valid
            samples and uses 0 to mark the ignored samples. Default: None.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    if isinstance(alpha, list):
        alpha = pred.new_tensor(alpha)
    pred_sigmoid = pred.sigmoid()
    target = target.type_as(pred)
    one_minus_pt = (1 - pred_sigmoid) * target + pred_sigmoid * (1 - target)
    focal_weight = (alpha * target + (1 - alpha) *
                    (1 - target)) * one_minus_pt.pow(gamma)

    loss = F.binary_cross_entropy_with_logits(
        pred, target, reduction='none') * focal_weight
    final_weight = torch.ones(1, pred.size(1)).type_as(loss)
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            # For most cases, weight is of shape (N, ),
            # which means it does not have the second axis num_class
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
    return loss


def sigmoid_focal_loss(pred,
                       target,
                       one_hot_target,
                       weight=None,
                       gamma=2.0,
                       alpha=0.5,
                       class_weight=None,
                       valid_mask=None,
                       reduction='mean',
                       avg_factor=None):
    r"""A warpper of cuda version `Focal Loss
    <https://arxiv.org/abs/1708.02002>`_.
    Args:
        pred (torch.Tensor): The prediction with shape (N, C), C is the number
            of classes.
        target (torch.Tensor): The learning label of the prediction. It's shape
            should be (N, )
        one_hot_target (torch.Tensor): The learning label with shape (N, C)
        weight (torch.Tensor, optional): Sample-wise loss weight.
        gamma (float, optional): The gamma for calculating the modulating
            factor. Defaults to 2.0.
        alpha (float | list[float], optional): A balanced form for Focal Loss.
            Defaults to 0.5.
        class_weight (list[float], optional): Weight of each class.
            Defaults to None.
        valid_mask (torch.Tensor, optional): A mask uses 1 to mark the valid
            samples and uses 0 to mark the ignored samples. Default: None.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and "sum".
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """
    # Function.apply does not accept keyword arguments, so the decorator
    # "weighted_loss" is not applicable
    final_weight = torch.ones(1, pred.size(1)).type_as(pred)
    if isinstance(alpha, list):
        # _sigmoid_focal_loss doesn't accept alpha of list type. Therefore, if
        # a list is given, we set the input alpha as 0.5. This means setting
        # equal weight for foreground class and background class. By
        # multiplying the loss by 2, the effect of setting alpha as 0.5 is
        # undone. The alpha of type list is used to regulate the loss in the
        # post-processing process.
        loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(),
                                   gamma, 0.5, None, 'none') * 2
        alpha = pred.new_tensor(alpha)
        final_weight = final_weight * (
            alpha * one_hot_target + (1 - alpha) * (1 - one_hot_target))
    else:
        loss = _sigmoid_focal_loss(pred.contiguous(), target.contiguous(),
                                   gamma, alpha, None, 'none')
    if weight is not None:
        if weight.shape != loss.shape and weight.size(0) == loss.size(0):
            # For most cases, weight is of shape (N, ),
            # which means it does not have the second axis num_class
            weight = weight.view(-1, 1)
        assert weight.dim() == loss.dim()
        final_weight = final_weight * weight
    if class_weight is not None:
        final_weight = final_weight * pred.new_tensor(class_weight)
    if valid_mask is not None:
        final_weight = final_weight * valid_mask
    loss = weight_reduce_loss(loss, final_weight, reduction, avg_factor)
    return loss


@LOSSES.register_module()
class FocalLoss(nn.Module):

    def __init__(self,
                 use_sigmoid=True,
                 gamma=2.0,
                 alpha=0.5,
                 reduction='mean',
                 class_weight=None,
                 loss_weight=1.0,
                 loss_name='loss_focal'):
        """`Focal Loss <https://arxiv.org/abs/1708.02002>`_
        Args:
            use_sigmoid (bool, optional): Whether to the prediction is
                used for sigmoid or softmax. Defaults to True.
            gamma (float, optional): The gamma for calculating the modulating
                factor. Defaults to 2.0.
            alpha (float | list[float], optional): A balanced form for Focal
                Loss. Defaults to 0.5. When a list is provided, the length
                of the list should be equal to the number of classes.
                Please be careful that this parameter is not the
                class-wise weight but the weight of a binary classification
                problem. This binary classification problem regards the
                pixels which belong to one class as the foreground
                and the other pixels as the background, each element in
                the list is the weight of the corresponding foreground class.
                The value of alpha or each element of alpha should be a float
                in the interval [0, 1]. If you want to specify the class-wise
                weight, please use `class_weight` parameter.
            reduction (str, optional): The method used to reduce the loss into
                a scalar. Defaults to 'mean'. Options are "none", "mean" and
                "sum".
            class_weight (list[float], optional): Weight of each class.
                Defaults to None.
            loss_weight (float, optional): Weight of loss. Defaults to 1.0.
            loss_name (str, optional): Name of the loss item. If you want this
                loss item to be included into the backward graph, `loss_` must
                be the prefix of the name. Defaults to 'loss_focal'.
        """
        super(FocalLoss, self).__init__()
        assert use_sigmoid is True, \
            'AssertionError: Only sigmoid focal loss supported now.'
        assert reduction in ('none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert isinstance(alpha, (float, list)), \
            'AssertionError: alpha should be of type float'
        assert isinstance(gamma, float), \
            'AssertionError: gamma should be of type float'
        assert isinstance(loss_weight, float), \
            'AssertionError: loss_weight should be of type float'
        assert isinstance(loss_name, str), \
            'AssertionError: loss_name should be of type str'
        assert isinstance(class_weight, list) or class_weight is None, \
            'AssertionError: class_weight must be None or of type list'
        self.use_sigmoid = use_sigmoid
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.class_weight = class_weight
        self.loss_weight = loss_weight
        self._loss_name = loss_name

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                ignore_index=255,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction with shape
                (N, C) where C = number of classes, or
                (N, C, d_1, d_2, ..., d_K) with K≥1 in the
                case of K-dimensional loss.
            target (torch.Tensor): The ground truth. If containing class
                indices, shape (N) where each value is 0≤targets[i]≤C−1,
                or (N, d_1, d_2, ..., d_K) with K≥1 in the case of
                K-dimensional loss. If containing class probabilities,
                same shape as the input.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to
                average the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used
                to override the original reduction method of the loss.
                Options are "none", "mean" and "sum".
            ignore_index (int, optional): The label index to be ignored.
                Default: 255
        Returns:
            torch.Tensor: The calculated loss
        """
        assert isinstance(ignore_index, int), \
            'ignore_index must be of type int'
        assert reduction_override in (None, 'none', 'mean', 'sum'), \
            "AssertionError: reduction should be 'none', 'mean' or " \
            "'sum'"
        assert pred.shape == target.shape or \
               (pred.size(0) == target.size(0) and
                pred.shape[2:] == target.shape[1:]), \
               "The shape of pred doesn't match the shape of target"

        original_shape = pred.shape

        # [B, C, d_1, d_2, ..., d_k] -> [C, B, d_1, d_2, ..., d_k]
        pred = pred.transpose(0, 1)
        # [C, B, d_1, d_2, ..., d_k] -> [C, N]
        pred = pred.reshape(pred.size(0), -1)
        # [C, N] -> [N, C]
        pred = pred.transpose(0, 1).contiguous()

        if original_shape == target.shape:
            # target with shape [B, C, d_1, d_2, ...]
            # transform it's shape into [N, C]
            # [B, C, d_1, d_2, ...] -> [C, B, d_1, d_2, ..., d_k]
            target = target.transpose(0, 1)
            # [C, B, d_1, d_2, ..., d_k] -> [C, N]
            target = target.reshape(target.size(0), -1)
            # [C, N] -> [N, C]
            target = target.transpose(0, 1).contiguous()
        else:
            # target with shape [B, d_1, d_2, ...]
            # transform it's shape into [N, ]
            target = target.view(-1).contiguous()
            valid_mask = (target != ignore_index).view(-1, 1)
            # avoid raising error when using F.one_hot()
            target = torch.where(target == ignore_index, target.new_tensor(0),
                                 target)

        reduction = (
            reduction_override if reduction_override else self.reduction)
        if self.use_sigmoid:
            num_classes = pred.size(1)
            if torch.cuda.is_available() and pred.is_cuda:
                if target.dim() == 1:
                    one_hot_target = F.one_hot(target, num_classes=num_classes)
                else:
                    one_hot_target = target
                    target = target.argmax(dim=1)
                    valid_mask = (target != ignore_index).view(-1, 1)
                calculate_loss_func = sigmoid_focal_loss
            else:
                one_hot_target = None
                if target.dim() == 1:
                    target = F.one_hot(target, num_classes=num_classes)
                else:
                    valid_mask = (target.argmax(dim=1) != ignore_index).view(
                        -1, 1)
                calculate_loss_func = py_sigmoid_focal_loss

            loss_cls = self.loss_weight * calculate_loss_func(
                pred,
                target,
                one_hot_target,
                weight,
                gamma=self.gamma,
                alpha=self.alpha,
                class_weight=self.class_weight,
                valid_mask=valid_mask,
                reduction=reduction,
                avg_factor=avg_factor)

            if reduction == 'none':
                # [N, C] -> [C, N]
                loss_cls = loss_cls.transpose(0, 1)
                # [C, N] -> [C, B, d1, d2, ...]
                # original_shape: [B, C, d1, d2, ...]
                loss_cls = loss_cls.reshape(original_shape[1],
                                            original_shape[0],
                                            *original_shape[2:])
                # [C, B, d1, d2, ...] -> [B, C, d1, d2, ...]
                loss_cls = loss_cls.transpose(0, 1).contiguous()
        else:
            raise NotImplementedError
        return loss_cls

    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name

# @LOSSES.register_module()
# class MultiFocalLoss(nn.Module):
#     def __init__(self, gamma = 2, alpha = 1, size_average = True, loss_weight=1.0, loss_name='loss_mfocal'):
#         super(MultiFocalLoss, self).__init__()
#         self.gamma = gamma
#         self.alpha = alpha
#         self.size_average = size_average
#         self.elipson = 0.000001
#         self.loss_weight=loss_weight
#         self._loss_name = loss_name

#     def forward(self, logits, labels, weight=None,ignore_index=None):
#         """
#         cal culates loss
#         logits: batch_size * labels_length * seq_length
#         labels: batch_size * seq_length
#         """
#         if labels.dim() > 2:
#             labels = labels.contiguous().view(labels.size(0), labels.size(1), -1)
#             labels = labels.transpose(1, 2)
#             labels = labels.contiguous().view(-1, labels.size(2)).squeeze()
#         if logits.dim() > 3:
#             logits = logits.contiguous().view(logits.size(0), logits.size(1), logits.size(2), -1)
#             logits = logits.transpose(2, 3)
#             logits = logits.contiguous().view(-1, logits.size(1), logits.size(3)).squeeze()
#         assert(logits.size(0) == labels.size(0))
#         assert(logits.size(2) == labels.size(1))
#         batch_size = logits.size(0)
#         labels_length = logits.size(1)
#         seq_length = logits.size(2)

#         # transpose labels into labels onehot
#         new_label = labels.unsqueeze(1)
#         label_onehot = torch.zeros([batch_size, labels_length, seq_length], device=new_label.device).scatter_(1, new_label, 1)
#         # label_onehot = label_onehot.permute(0, 2, 1) # transpose, batch_size * seq_length * labels_length

#         # calculate log
#         log_p = F.log_softmax(logits, dim=1)
#         pt = label_onehot * log_p
#         sub_pt = 1 - pt
#         fl = -self.alpha * (sub_pt)**self.gamma * log_p
#         wfl= self.loss_weight * fl
#         # if self.size_average:
#         #     return fl.mean()
#         # else:
#         #     return fl.sum()
#         if self.size_average:
#             return wfl.mean()
#         else:
#             return wfl.sum()
        
#     @property
#     def loss_name(self):
#         """Loss Name.

#         This function must be implemented and will return the name of this
#         loss function. This name will be used to combine different loss items
#         by simple sum operation. In addition, if you want this loss item to be
#         included into the backward graph, `loss_` must be the prefix of the
#         name.
#         Returns:
#             str: The name of this loss item.
#         """
#         return self._loss_name
    

# @LOSSES.register_module()
# class MultiFocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=2, device='cpu',loss_weight=1.0, loss_name='loss_mfocal'):
#         super(MultiFocalLoss, self).__init__()
#         # focusing hyper-parameter gamma
#         self.gamma = gamma

#         # class weights will act as the alpha parameter
#         self.weight = weight
        
#         # using deivce (cpu or gpu)
#         self.device = device
        
#         self.ce_loss = nn.CrossEntropyLoss()

#         self.loss_weight=loss_weight
#         self._loss_name = loss_name

#     def forward(self, logits, labels, weight=None,ignore_index=None):
#         focal_loss = 0

#         for i in range(len(logits)):#for every data in a batch
#             # -log(pt)
#             cur_ce_loss = self.ce_loss(logits[i].view(-1, logits[i].size()[-1]), labels[i].view(-1))#the calculation of CE has something wrong
#             # pt
#             pt = torch.exp(-cur_ce_loss)

#             if self.weight is not None:
#                 # alpha * (1-pt)^gamma * -log(pt)
#                 cur_focal_loss = self.weight[labels[i]] * ((1 - pt) ** self.gamma) * cur_ce_loss
#             else:
#                 # (1-pt)^gamma * -log(pt)
#                 cur_focal_loss = ((1 - pt) ** self.gamma) * cur_ce_loss
                
#             focal_loss = focal_loss + cur_focal_loss

#         if self.weight is not None:
#             focal_loss = focal_loss / self.weight.sum()
#             return focal_loss.to(self.device)
        
#         focal_loss = self.loss_weight * focal_loss / torch.tensor(len(logits))    
#         return focal_loss.to(self.device)
        
#     @property
#     def loss_name(self):
#         """Loss Name.

#         This function must be implemented and will return the name of this
#         loss function. This name will be used to combine different loss items
#         by simple sum operation. In addition, if you want this loss item to be
#         included into the backward graph, `loss_` must be the prefix of the
#         name.
#         Returns:
#             str: The name of this loss item.
#         """
#         return self._loss_name

@LOSSES.register_module()
class MultiFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean',loss_weight=1.0,loss_name="loss_mfocal"):
        """Focal Loss
        
        Args:
            alpha (Tensor/list/float, optional): 
            gamma (float, optional): 
            reduction (str, optional): 'mean'/'sum'/'none'
        """
        super(MultiFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.loss_weight=loss_weight
        self._loss_name = loss_name

    def forward(self, inputs, targets,weight=None,ignore_index=None):
        assert inputs.dim() == targets.dim() + 1, "error"
        
        num_classes = inputs.size(1) 
        
        if self.alpha is None:
            alpha = torch.ones(num_classes, device=inputs.device, dtype=inputs.dtype)
        elif isinstance(self.alpha, (list, np.ndarray)):
            alpha = torch.tensor(self.alpha, device=inputs.device, dtype=inputs.dtype)
        elif isinstance(self.alpha, (float, int)):
            alpha = torch.full((num_classes,), self.alpha, 
                              device=inputs.device, dtype=inputs.dtype)
        else:
            alpha = self.alpha.to(device=inputs.device, dtype=inputs.dtype)
        
        assert alpha.shape[0] == num_classes, 'the length of alpha should be the same as categories'

        log_softmax = F.log_softmax(inputs, dim=1)
        
        # 收集对应类别的log概率
        targets_view = targets.unsqueeze(1)  # 增加通道维度
        log_pt = torch.gather(log_softmax, dim=1, index=targets_view).squeeze(1)
        
        # 计算交叉熵损失
        ce_loss = -log_pt
        
        # 计算概率值pt
        pt = torch.exp(log_pt)
        
        # 计算调制因子
        modulating_factor = (1 - pt) ** self.gamma
        
        # 获取类别权重
        alpha_weight = alpha[targets.flatten()].view_as(targets)
        
        # 计算最终损失
        focal_loss = alpha_weight * modulating_factor * ce_loss
        
        # 聚合损失
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.
        Returns:
            str: The name of this loss item.
        """
        return self._loss_name