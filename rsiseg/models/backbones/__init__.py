# Copyright (c) OpenMMLab. All rights reserved.
#from .mit import MixVisionTransformer, mit_b5
from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnet_pretrain import ResNetPretrain
__all__ = [
    'ResNet', 'ResNetV1c','ResNetPretrain',
]
