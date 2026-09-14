# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_segmentor, init_segmentor, show_result_pyplot
from .test import multi_gpu_test, single_gpu_test, single_gpu_test_fitbs, single_gpu_test_org, single_gpu_test1
from .train import (get_root_logger, init_random_seed, set_random_seed,
                    train_segmentor, train_segmentor_new, train_segmentor_sourceonly)

__all__ = [
    'get_root_logger', 'set_random_seed', 'train_segmentor', 'train_segmentor_new','train_segmentor_sourceonly','init_segmentor','train_segmentor_org','train_segmentor_sepico','train_segmentor1','train_segmentorSTDASegNet','train_segmentor_DAFormer',
    'inference_segmentor', 'multi_gpu_test', 'single_gpu_test','single_gpu_test_org','single_gpu_test1','single_gpu_test_sepico','single_gpu_test_STDASegNet','single_gpu_test_tbs',
    'show_result_pyplot', 'init_random_seed','train_segmentor_fitbs','single_gpu_test_fitbs'
]
