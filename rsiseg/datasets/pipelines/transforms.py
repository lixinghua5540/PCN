import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random
import pdb
import skimage
import copy
from ..builder import PIPELINES
import cv2

@PIPELINES.register_module()
class Resize(object):
    """Resize images & seg.

    This transform resizes the input image to some scale. If the input dict
    contains the key "scale", then the scale in the input dict is used,
    otherwise the specified scale in the init method is used.

    ``img_scale`` can be Nong, a tuple (single-scale) or a list of tuple
    (multi-scale). There are 4 multiscale modes:

    - ``ratio_range is not None``:
    1. When img_scale is None, img_scale is the shape of image in results
        (img_scale = results['img'].shape[:2]) and the image is resized based
        on the original size. (mode 1)
    2. When img_scale is a tuple (single-scale), randomly sample a ratio from
        the ratio range and multiply it with the image scale. (mode 2)

    - ``ratio_range is None and multiscale_mode == "range"``: randomly sample a
    scale from the a range. (mode 3)

    - ``ratio_range is None and multiscale_mode == "value"``: randomly sample a
    scale from multiple scales. (mode 4)

    Args:
        img_scale (tuple or list[tuple]): Images scales for resizing.
        multiscale_mode (str): Either "range" or "value".
        ratio_range (tuple[float]): (min_ratio, max_ratio)
        keep_ratio (bool): Whether to keep the aspect ratio when resizing the
            image.
    """

    def __init__(self,
                 img_scale=None,
                 B_img_scale=None,
                 multiscale_mode='range',
                 ratio_range=None,
                 keep_ratio=True):
        if img_scale is None:
            self.img_scale = None
        else:
            if isinstance(img_scale, list):
                self.img_scale = img_scale
            else:
                self.img_scale = [img_scale]
            assert mmcv.is_list_of(self.img_scale, tuple)

        if ratio_range is not None:
            # mode 1: given img_scale=None and a range of image ratio
            # mode 2: given a scale and a range of image ratio
            assert self.img_scale is None or len(self.img_scale) == 1
        else:
            # mode 3 and 4: given multiple scales or a range of scales
            assert multiscale_mode in ['value', 'range']

        ## added by LYU: 2022/03/24
        if B_img_scale is None:
            self.B_img_scale = None
        else:
            if isinstance(B_img_scale, list):
                self.B_img_scale = B_img_scale
            else:
                self.B_img_scale = [B_img_scale]
            assert mmcv.is_list_of(self.B_img_scale, tuple)
        self.multiscale_mode = multiscale_mode
        self.ratio_range = ratio_range
        self.keep_ratio = keep_ratio

    @staticmethod
    def random_select(img_scales):
        """Randomly select an img_scale from given candidates.

        Args:
            img_scales (list[tuple]): Images scales for selection.

        Returns:
            (tuple, int): Returns a tuple ``(img_scale, scale_dix)``,
                where ``img_scale`` is the selected image scale and
                ``scale_idx`` is the selected index in the given candidates.
        """

        assert mmcv.is_list_of(img_scales, tuple)
        scale_idx = np.random.randint(len(img_scales))
        img_scale = img_scales[scale_idx]
        return img_scale, scale_idx

    @staticmethod
    def random_sample(img_scales):
        """Randomly sample an img_scale when ``multiscale_mode=='range'``.

        Args:
            img_scales (list[tuple]): Images scale range for sampling.
                There must be two tuples in img_scales, which specify the lower
                and uper bound of image scales.

        Returns:
            (tuple, None): Returns a tuple ``(img_scale, None)``, where
                ``img_scale`` is sampled scale and None is just a placeholder
                to be consistent with :func:`random_select`.
        """

        assert mmcv.is_list_of(img_scales, tuple) and len(img_scales) == 2
        img_scale_long = [max(s) for s in img_scales]
        img_scale_short = [min(s) for s in img_scales]
        long_edge = np.random.randint(
            min(img_scale_long),
            max(img_scale_long) + 1)
        short_edge = np.random.randint(
            min(img_scale_short),
            max(img_scale_short) + 1)
        img_scale = (long_edge, short_edge)
        return img_scale, None

    @staticmethod
    def random_sample_ratio(img_scale, ratio_range):
        """Randomly sample an img_scale when ``ratio_range`` is specified.

        A ratio will be randomly sampled from the range specified by
        ``ratio_range``. Then it would be multiplied with ``img_scale`` to
        generate sampled scale.

        Args:
            img_scale (tuple): Images scale base to multiply with ratio.
            ratio_range (tuple[float]): The minimum and maximum ratio to scale
                the ``img_scale``.

        Returns:
            (tuple, None): Returns a tuple ``(scale, None)``, where
                ``scale`` is sampled ratio multiplied with ``img_scale`` and
                None is just a placeholder to be consistent with
                :func:`random_select`.
        """

        assert isinstance(img_scale, tuple) and len(img_scale) == 2
        min_ratio, max_ratio = ratio_range
        assert min_ratio <= max_ratio
        ratio = np.random.random_sample() * (max_ratio - min_ratio) + min_ratio
        scale = int(img_scale[0] * ratio), int(img_scale[1] * ratio)
        return scale, None

    def _random_scale(self, results):
        """Randomly sample an img_scale according to ``ratio_range`` and
        ``multiscale_mode``.

        If ``ratio_range`` is specified, a ratio will be sampled and be
        multiplied with ``img_scale``.
        If multiple scales are specified by ``img_scale``, a scale will be
        sampled according to ``multiscale_mode``.
        Otherwise, single scale will be used.

        Args:
            results (dict): Result dict from :obj:`dataset`.

        Returns:
            dict: Two new keys 'scale` and 'scale_idx` are added into
                ``results``, which would be used by subsequent pipelines.
        """

        if self.ratio_range is not None:
            if self.img_scale is None:
                h, w = results['img'].shape[:2]
                scale, scale_idx = self.random_sample_ratio((w, h),
                                                            self.ratio_range)
            else:
                scale, scale_idx = self.random_sample_ratio(
                    self.img_scale[0], self.ratio_range)
        elif len(self.img_scale) == 1:
            scale, scale_idx = self.img_scale[0], 0
        elif self.multiscale_mode == 'range':
            scale, scale_idx = self.random_sample(self.img_scale)
        elif self.multiscale_mode == 'value':
            scale, scale_idx = self.random_select(self.img_scale)
        else:
            raise NotImplementedError

        results['scale'] = scale
        results['scale_idx'] = scale_idx
        ## added by LYU: 2022/03/24
        if self.B_img_scale is not None:
            results['B_scale'], results['B_scale_idx'] = self.B_img_scale[0], 0

    def _resize_img(self, results):
        """Resize images with ``results['scale']``."""
        if self.keep_ratio:
            img, scale_factor = mmcv.imrescale(
                results['img'], results['scale'], return_scale=True)
            # the w_scale and h_scale has minor difference
            # a real fix should be done in the mmcv.imrescale in the future
            new_h, new_w = img.shape[:2]
            h, w = results['img'].shape[:2]
            w_scale = new_w / w
            h_scale = new_h / h
        else:
            img, w_scale, h_scale = mmcv.imresize(
                results['img'], results['scale'], return_scale=True)
        scale_factor = np.array([w_scale, h_scale, w_scale, h_scale],
                                dtype=np.float32)
        results['img'] = img
        results['img_shape'] = img.shape
        results['pad_shape'] = img.shape  # in case that there is no padding
        results['scale_factor'] = scale_factor
        results['keep_ratio'] = self.keep_ratio
        ## added by LYU: 2022/03/24
        if results.get('B_img') is not None:
            B_img, B_w_scale, B_h_scale = mmcv.imresize(
                results['B_img'], results['B_scale'], return_scale=True)
            B_scale_factor = np.array([B_w_scale, B_h_scale, B_w_scale, B_h_scale],
                                dtype=np.float32)
            results['B_img'] = B_img
            results['B_img_shape'] = B_img.shape
            results['B_pad_shape'] = B_img.shape  # in case that there is no padding
            results['B_scale_factor'] = B_scale_factor

    def _resize_seg(self, results):
        """Resize semantic segmentation map with ``results['scale']``."""
        for key in results.get('seg_fields', []):
            if self.keep_ratio:
                gt_seg = mmcv.imrescale(
                    results[key], results['scale'], interpolation='nearest')
            else:
                gt_seg = mmcv.imresize(
                    results[key], results['scale'], interpolation='nearest')
            results[key] = gt_seg

    def _split_resize(self, data, scale, interpolation='nearest'):
        H, W, C = data.shape
        MAX_C = 512
        if C > MAX_C:
            splits = []
            for i in range((C-1) // MAX_C + 1):
                cur_split = mmcv.imresize(data[:, :, i*MAX_C:(i+1)*MAX_C], scale, interpolation=interpolation)
                splits.append(cur_split)
            result = np.concatenate(splits, axis=2)
        else:
            result = mmcv.imresize(data, scale, interpolation=interpolation)

        return result

    def _resize_feat(self, results):
        if 'feats' in results:
            pass
            # feats = skimage.transform.resize(results['feats'], results['scale'], order=0)
            # feats = self._split_resize(results['feats'], results['scale'], interpolation='nearest')
            # results['feats'] = feats


    def __call__(self, results):
        """Call function to resize images, bounding boxes, masks, semantic
        segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Resized results, 'img_shape', 'pad_shape', 'scale_factor',
                'keep_ratio' keys are added into result dict.
        """

        if 'scale' not in results:
            self._random_scale(results)
        self._resize_img(results)
        self._resize_seg(results)
        self._resize_feat(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(img_scale={self.img_scale}, '
                     f'multiscale_mode={self.multiscale_mode}, '
                     f'ratio_range={self.ratio_range}, '
                     f'keep_ratio={self.keep_ratio})')
        return repr_str


@PIPELINES.register_module()
class RandomFlip(object):
    """Flip the image & seg.

    If the input dict contains the key "flip", then the flag will be used,
    otherwise it will be randomly decided by a ratio specified in the init
    method.

    Args:
        prob (float, optional): The flipping probability. Default: None.
        direction(str, optional): The flipping direction. Options are
            'horizontal' and 'vertical'. Default: 'horizontal'.
    """

    @deprecated_api_warning({'flip_ratio': 'prob'}, cls_name='RandomFlip')
    def __init__(self, prob=None, direction='horizontal'):
        self.prob = prob
        self.direction = direction
        if prob is not None:
            assert prob >= 0 and prob <= 1
        assert direction in ['horizontal', 'vertical']

    def __call__(self, results):
        """Call function to flip bounding boxes, masks, semantic segmentation
        maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Flipped results, 'flip', 'flip_direction' keys are added into
                result dict.
        """

        flip = True if np.random.rand() < self.prob else False
        if not 'flip' in results:
            results['flip'] = flip
        else:
            results['flip'] = results['flip'] or flip

        # results[f'flip_{self.direction}'] = flip

        if flip:

            if 'flip_direction' not in results:
                results['flip_direction'] = [self.direction]
            else:
                results['flip_direction'].append(self.direction)

            # flip image
            results['img'] = mmcv.imflip(
                results['img'], direction=self.direction)

            # flip segs
            for key in results.get('seg_fields', []):
                # use copy() to make numpy stride positive
                results[key] = mmcv.imflip(
                    results[key], direction=self.direction).copy()

            if 'feats' in results:
                pass
                # mmcv.imflip(results['feats'], direction=results['flip_direction']).copy()

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(prob={self.prob})'

@PIPELINES.register_module()
class Pad2(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        if self.size is not None:
            padded_img = mmcv.impad(
                results['img'], shape=self.size, pad_val=self.pad_val)
        elif self.size_divisor is not None:
            padded_img = mmcv.impad_to_multiple(
                results['img'], self.size_divisor, pad_val=self.pad_val)
        results['img'] = padded_img
        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key],
                shape=results['pad_shape'][:2],
                pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str
    
@PIPELINES.register_module()
class Pad(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""
        for img_key in results['img_fields']:
            if self.size is not None:
                padded_img = mmcv.impad(
                    results[img_key], shape=self.size, pad_val=self.pad_val)
            elif self.size_divisor is not None:
                padded_img = mmcv.impad_to_multiple(
                    results[img_key], self.size_divisor, pad_val=self.pad_val)
            results[img_key] = padded_img

        results['pad_shape'] = padded_img.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key],
                shape=results['pad_shape'][:2],
                pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class Normalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        # added by LYU: 2022/03/25
        if results.get('B_img') is not None:
            results['B_img'] = mmcv.imnormalize(results['B_img'], self.mean, self.std,
                                          self.to_rgb)
        
        for img_key in results['img_fields']:#confirm the img_fields do not have B_img, but it does not have img?
            results[img_key] = mmcv.imnormalize(
                results[img_key], self.mean, self.std, self.to_rgb)

        if 'ori_img' in results:
            results['ori_img'] = mmcv.imnormalize(
                results['ori_img'], self.mean, self.std, self.to_rgb)

        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str

@PIPELINES.register_module()
class Normalize2(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        results['img'] = mmcv.imnormalize(results['img'], self.mean, self.std,
                                          self.to_rgb)
        results['img_norm_cfg'] = dict(
            mean=self.mean, std=self.std, to_rgb=self.to_rgb)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str
    
@PIPELINES.register_module()
class Normalize16bit:
    """Normalize 16-bit multi-channel images with dynamic channel support
    
    Args:
        mean (sequence): Per-channel mean values
        std (sequence): Per-channel std values
        to_rgb (bool): Convert BGR to RGB (only for 3/4-channel images)
        bit_depth (int): Input bit depth (16 or 32)
        rescale (bool): Scale back to original bit depth after normalization
    """

    def __init__(self, mean, std, to_rgb=True, 
                 bit_depth=16, rescale=False):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb
        self.bit_depth = bit_depth
        self.rescale = rescale
        
        # 验证参数合法性
        if bit_depth not in [16, 32]:
            raise ValueError("bit_depth must be 16 or 32")
        if len(mean) != len(std):
            raise ValueError("Mean and std must have same length")

    def _convert_dtype(self, img):
        """处理不同位深度的数据类型转换"""
        if self.bit_depth == 16:
            # if img.dtype != np.uint16:
            #     img = img.astype(np.uint16)#data type has something wrong
            # return img.astype(np.float32) #/ 65535.0  # 归一化到0-1
            
            if img.dtype != np.float32:
                img = img.astype(np.float32)#data type has something wrong
            return img.astype(np.float32)
        return img.astype(np.float32)  # 32位数据直接转换

    def _validate_channels(self, img):
        """通道数一致性检查"""
        input_channels = 1 if img.ndim == 2 else img.shape[2]
        if len(self.mean) != input_channels:
            raise ValueError(
                f"Input has {input_channels} channels but "
                f"normalization expects {len(self.mean)} channels"
            )

    def __call__(self, results):
        # 统一数据预处理流程
        def process_img(img):
            img = self._convert_dtype(img)
            
            self._validate_channels(img)
            
            # BGR to RGB
            # if self.to_rgb:
            #     if img.ndim == 3:
            #         if img.shape[2] == 3:  # 标准RGB
            #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #         elif img.shape[2] == 4:  # RGBA
            #             img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGBA)
            
            # 执行归一化
            img = (img - self.mean) / self.std
            
            # 可选：恢复原始位深度
            if self.rescale:
                if self.bit_depth == 16:
                    img = (img * 65535).clip(0, 65535).astype(np.uint16)
                else:
                    img = img.astype(np.float32)
                    
            return img

        # 处理主图像字段
        for img_key in results['img_fields']:
            if img_key in results:
                results[img_key] = process_img(results[img_key])
        
        # 处理特殊字段
        special_fields = ['B_img', 'ori_img']
        for field in special_fields:
            if results.get(field) is not None:
                results[field] = process_img(results[field])

        # 保存归一化配置
        results['img_norm_cfg'] = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'to_rgb': self.to_rgb,
            'bit_depth': self.bit_depth
        }

        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(mean={self.mean}, std={self.std}, '
                f'to_rgb={self.to_rgb}, bit_depth={self.bit_depth}, '
                f'rescale={self.rescale})')

@PIPELINES.register_module()
class PercentileNormalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, lower, upper, axis=None):
        self.lower = lower
        self.upper = upper
        self.axis = axis

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """

        assert self.lower < self.upper
        lower_percentile = np.percentile(results['img'], self.lower, axis=self.axis)
        upper_percentile = np.percentile(results['img'], self.upper, axis=self.axis)
        img_normalized = np.clip(
            (results['img'] - lower_percentile) / (upper_percentile - lower_percentile + 1e-8), 0, 1
        )
        results['img'] = img_normalized.astype(np.float32)

        return results



    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.lower}, std={self.upper}, axis=' \
                    f'{self.axis})'
        return repr_str

@PIPELINES.register_module()
class MultiDomainClipNormalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, norm_dict, key_name, to_rgb=True, axis=None):
        self.norm_dict = norm_dict
        self.key_name = key_name
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        mean = self.norm_dict[results[self.key_name]]['mean']
        mean = np.array(mean).reshape(1, 1, -1).astype(np.float32)

        std = self.norm_dict[results[self.key_name]]['std']
        std = np.array(std).reshape(1, 1, -1).astype(np.float32)

        min_value = mean - 3 * std
        max_value = mean + 3 * std

        img = results['img']
        img = (img.astype(np.float32) - min_value) / (max_value - min_value)
        img = np.clip(img, 0, 1)

        if self.to_rgb:
            img = img[:, :, [2,1,0]]

        results['img'] = img

        return results






@PIPELINES.register_module()
class Rerange(object):
    """Rerange the image pixel value.

    Args:
        min_value (float or int): Minimum value of the reranged image.
            Default: 0.
        max_value (float or int): Maximum value of the reranged image.
            Default: 255.
    """

    def __init__(self, min_value=0, max_value=255):
        assert isinstance(min_value, float) or isinstance(min_value, int)
        assert isinstance(max_value, float) or isinstance(max_value, int)
        assert min_value < max_value
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, results):
        """Call function to rerange images.

        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Reranged results.
        """

        img = results['img']
        img_min_value = np.min(img)
        img_max_value = np.max(img)

        assert img_min_value < img_max_value
        # rerange to [0, 1]
        img = (img - img_min_value) / (img_max_value - img_min_value)
        # rerange to [min_value, max_value]
        img = img * (self.max_value - self.min_value) + self.min_value
        results['img'] = img

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(min_value={self.min_value}, max_value={self.max_value})'
        return repr_str


@PIPELINES.register_module()
class CLAHE(object):
    """Use CLAHE method to process the image.

    See `ZUIDERVELD,K. Contrast Limited Adaptive Histogram Equalization[J].
    Graphics Gems, 1994:474-485.` for more information.

    Args:
        clip_limit (float): Threshold for contrast limiting. Default: 40.0.
        tile_grid_size (tuple[int]): Size of grid for histogram equalization.
            Input image will be divided into equally sized rectangular tiles.
            It defines the number of tiles in row and column. Default: (8, 8).
    """

    def __init__(self, clip_limit=40.0, tile_grid_size=(8, 8)):
        assert isinstance(clip_limit, (float, int))
        self.clip_limit = clip_limit
        assert is_tuple_of(tile_grid_size, int)
        assert len(tile_grid_size) == 2
        self.tile_grid_size = tile_grid_size

    def __call__(self, results):
        """Call function to Use CLAHE method process images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        for i in range(results['img'].shape[2]):
            results['img'][:, :, i] = mmcv.clahe(
                np.array(results['img'][:, :, i], dtype=np.uint8),
                self.clip_limit, self.tile_grid_size)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(clip_limit={self.clip_limit}, '\
                    f'tile_grid_size={self.tile_grid_size})'
        return repr_str


@PIPELINES.register_module()
class RandomCrop(object):
    """Random crop the image & seg.

    Args:
        crop_size (tuple): Expected size after cropping, (h, w).
        cat_max_ratio (float): The maximum ratio that single category could
            occupy.
    """

    def __init__(self, crop_size, cat_max_ratio=1., ignore_index=255):
        assert crop_size[0] > 0 and crop_size[1] > 0
        self.crop_size = crop_size
        self.cat_max_ratio = cat_max_ratio
        self.ignore_index = ignore_index

    def get_crop_bbox(self, img):
        """Randomly get a crop bounding box."""
        margin_h = max(img.shape[0] - self.crop_size[0], 0)
        margin_w = max(img.shape[1] - self.crop_size[1], 0)
        offset_h = np.random.randint(0, margin_h + 1)
        offset_w = np.random.randint(0, margin_w + 1)
        crop_y1, crop_y2 = offset_h, offset_h + self.crop_size[0]
        crop_x1, crop_x2 = offset_w, offset_w + self.crop_size[1]

        return crop_y1, crop_y2, crop_x1, crop_x2

    def crop(self, img, crop_bbox):
        """Crop from ``img``"""
        crop_y1, crop_y2, crop_x1, crop_x2 = crop_bbox
        img = img[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return img

    def proportional_crop(self, data, crop_bbox, ratio):
        """Crop from ``img``"""
        rescale = lambda x: int(x * ratio)
        crop_y1, crop_y2, crop_x1, crop_x2 = map(rescale, crop_bbox)
        data = data[crop_y1:crop_y2, crop_x1:crop_x2, ...]
        return data

    def __call__(self, results):
        """Call function to randomly crop images, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Randomly cropped results, 'img_shape' key in result dict is
                updated according to crop size.
        """

        img = results['img']
        ori_shape = img.shape
        crop_bbox = self.get_crop_bbox(img)
        if self.cat_max_ratio < 1.:
            # Repeat 10 times
            for _ in range(10):
                seg_temp = self.crop(results['gt_semantic_seg'], crop_bbox)
                labels, cnt = np.unique(seg_temp, return_counts=True)
                cnt = cnt[labels != self.ignore_index]
                if len(cnt) > 1 and np.max(cnt) / np.sum(
                        cnt) < self.cat_max_ratio:
                    break
                crop_bbox = self.get_crop_bbox(img)

        # crop the image
        img = self.crop(img, crop_bbox)
        img_shape = img.shape
        results['img'] = img
        results['img_shape'] = img_shape

        # crop semantic seg
        for key in results.get('seg_fields', []):
            results[key] = self.crop(results[key], crop_bbox)

        results['crop_bbox'] = crop_bbox

        if 'feats' in results:
            pass
            # crop the feature map proportionally
            # feats = results['feats']
            # h, w, c = feats.shape
            # ratio = h / ori_shape[0]
            # feats = self.proportional_crop(feats, crop_bbox, ratio)
            # feats = self.crop(feats, crop_bbox)
            # results['feats'] = feats
        if results.get('B_img') is not None:#B image has no label, so we can not calculate the maximum number of spedific classes, for STDASegNet
            B_img = results['B_img']
            # random brightness
            crop_bbox = self.get_crop_bbox(B_img)
            # crop the image
            B_img = self.crop(B_img, crop_bbox) 
            results['B_img'] = B_img

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(crop_size={self.crop_size})'


@PIPELINES.register_module()
class RandomRotate(object):
    """Rotate the image & seg.

    Args:
        prob (float): The rotation probability.
        degree (float, tuple[float]): Range of degrees to select from. If
            degree is a number instead of tuple like (min, max),
            the range of degree will be (``-degree``, ``+degree``)
        pad_val (float, optional): Padding value of image. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
        center (tuple[float], optional): Center point (w, h) of the rotation in
            the source image. If not specified, the center of the image will be
            used. Default: None.
        auto_bound (bool): Whether to adjust the image size to cover the whole
            rotated image. Default: False
    """

    def __init__(self,
                 prob,
                 degree,
                 pad_val=0,
                 seg_pad_val=255,
                 center=None,
                 auto_bound=False):
        self.prob = prob
        assert prob >= 0 and prob <= 1
        if isinstance(degree, (float, int)):
            assert degree > 0, f'degree {degree} should be positive'
            self.degree = (-degree, degree)
        else:
            self.degree = degree
        assert len(self.degree) == 2, f'degree {self.degree} should be a ' \
                                      f'tuple of (min, max)'
        self.pal_val = pad_val
        self.seg_pad_val = seg_pad_val
        self.center = center
        self.auto_bound = auto_bound

    def __call__(self, results):
        """Call function to rotate image, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Rotated results.
        """

        rotate = True if np.random.rand() < self.prob else False
        degree = np.random.uniform(min(*self.degree), max(*self.degree))
        if rotate:
            # rotate image
            results['img'] = mmcv.imrotate(
                results['img'],
                angle=degree,
                border_value=self.pal_val,
                center=self.center,
                auto_bound=self.auto_bound)

            # rotate segs
            for key in results.get('seg_fields', []):
                results[key] = mmcv.imrotate(
                    results[key],
                    angle=degree,
                    border_value=self.seg_pad_val,
                    center=self.center,
                    auto_bound=self.auto_bound,
                    interpolation='nearest')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(prob={self.prob}, ' \
                    f'degree={self.degree}, ' \
                    f'pad_val={self.pal_val}, ' \
                    f'seg_pad_val={self.seg_pad_val}, ' \
                    f'center={self.center}, ' \
                    f'auto_bound={self.auto_bound})'
        return repr_str


@PIPELINES.register_module()
class RGB2Gray(object):
    """Convert RGB image to grayscale image.

    This transform calculate the weighted mean of input image channels with
    ``weights`` and then expand the channels to ``out_channels``. When
    ``out_channels`` is None, the number of output channels is the same as
    input channels.

    Args:
        out_channels (int): Expected number of output channels after
            transforming. Default: None.
        weights (tuple[float]): The weights to calculate the weighted mean.
            Default: (0.299, 0.587, 0.114).
    """

    def __init__(self, out_channels=None, weights=(0.299, 0.587, 0.114)):
        assert out_channels is None or out_channels > 0
        self.out_channels = out_channels
        assert isinstance(weights, tuple)
        for item in weights:
            assert isinstance(item, (float, int))
        self.weights = weights

    def __call__(self, results):
        """Call function to convert RGB image to grayscale image.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with grayscale image.
        """
        img = results['img']
        assert len(img.shape) == 3
        assert img.shape[2] == len(self.weights)
        weights = np.array(self.weights).reshape((1, 1, -1))
        img = (img * weights).sum(2, keepdims=True)
        if self.out_channels is None:
            img = img.repeat(weights.shape[2], axis=2)
        else:
            img = img.repeat(self.out_channels, axis=2)

        results['img'] = img
        results['img_shape'] = img.shape

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(out_channels={self.out_channels}, ' \
                    f'weights={self.weights})'
        return repr_str


@PIPELINES.register_module()
class AdjustGamma(object):
    """Using gamma correction to process the image.

    Args:
        gamma (float or int): Gamma value used in gamma correction.
            Default: 1.0.
    """

    def __init__(self, gamma=1.0):
        assert isinstance(gamma, float) or isinstance(gamma, int)
        assert gamma > 0
        self.gamma = gamma
        inv_gamma = 1.0 / gamma
        self.table = np.array([(i / 255.0)**inv_gamma * 255
                               for i in np.arange(256)]).astype('uint8')

    def __call__(self, results):
        """Call function to process the image with gamma correction.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Processed results.
        """

        results['img'] = mmcv.lut_transform(
            np.array(results['img'], dtype=np.uint8), self.table)

        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(gamma={self.gamma})'


@PIPELINES.register_module()
class SegRescale(object):
    """Rescale semantic segmentation maps.

    Args:
        scale_factor (float): The scale factor of the final output.
    """

    def __init__(self, scale_factor=1):
        self.scale_factor = scale_factor

    def __call__(self, results):
        """Call function to scale the semantic segmentation map.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with semantic segmentation map scaled.
        """
        for key in results.get('seg_fields', []):
            if self.scale_factor != 1:
                results[key] = mmcv.imrescale(
                    results[key], self.scale_factor, interpolation='nearest')
        return results

    def __repr__(self):
        return self.__class__.__name__ + f'(scale_factor={self.scale_factor})'


@PIPELINES.register_module()
class PhotoMetricDistortion(object):
    """Apply photometric distortion to image sequentially, every transformation
    is applied with a probability of 0.5. The position of random contrast is in
    second or second to last.

    1. random brightness
    2. random contrast (mode 0)
    3. convert color from BGR to HSV
    4. random saturation
    5. random hue
    6. convert color from HSV to BGR
    7. random contrast (mode 1)
    8. randomly swap channels

    Args:
        brightness_delta (int): delta of brightness.
        contrast_range (tuple): range of contrast.
        saturation_range (tuple): range of saturation.
        hue_delta (int): delta of hue.
    """

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)#saturation has something wrong for 4 channels
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img'] = img

        # added by LYU: 2022/03/25
        if results.get('B_img') is not None:
            B_img = results['B_img']
            # random brightness
            B_img = self.brightness(B_img)
            
            # mode == 0 --> do random contrast first
            # mode == 1 --> do random contrast last
            mode = random.randint(2)
            if mode == 1:
                B_img = self.contrast(B_img)
                
            # random saturation
            B_img = self.saturation(B_img)
            
            # random hue
            B_img = self.hue(B_img)
            
            # random contrast
            if mode == 0:
                B_img = self.contrast(B_img)
            results['B_img'] = B_img
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str
    
@PIPELINES.register_module()
class PhotoMetricDistortion4C_DataChannel(object):
    """支持四通道影像的光度畸变（第四个通道为数据通道，不做任何处理）"""
    
    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    # def convert(self, img, alpha=1, beta=0):
    #     """处理前三个通道的线性变换（第四通道不变）"""
    #     # 只对前三个通道做计算
    #     rgb = img[..., :3].astype(np.float32) * alpha + beta
    #     rgb = np.clip(rgb, 0, 255).astype(np.uint8)
        
    #     # 保持第四通道不变
    #     if img.shape[-1] == 4:
    #         return np.concatenate([rgb, img[..., [3]]], axis=-1)
    #     return rgb
    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        #img = np.clip(img, 0, 255)
        return img#.astype(np.uint8)
    def _split_data_channel(self, img):
        """分离数据通道（假设输入是RGBN或其他四通道格式）"""
        if img.shape[-1] == 4:
            rgb, data = img[..., :3], img[..., 3:]
            return rgb, data
        return img, None

    def _merge_data_channel(self, rgb, data):
        """合并处理后的RGB和数据通道"""
        if data is not None:
            return np.concatenate([rgb, data], axis=-1)
        return rgb

    # def brightness(self, img):
    #     """仅调整前三个通道的亮度"""
    #     rgb, data = self._split_data_channel(img)
    #     if random.randint(2):
    #         delta = random.uniform(-self.brightness_delta, self.brightness_delta)
    #         # 使用convert方法保持第四通道不变
    #         return self.convert(img, beta=delta)
    #     return img

    # def contrast(self, img):
    #     """仅调整前三个通道的对比度"""
    #     rgb, data = self._split_data_channel(img)
    #     if random.randint(2):
    #         alpha = random.uniform(self.contrast_lower, self.contrast_upper)
    #         # 使用convert方法保持第四通道不变
    #         return self.convert(img, alpha=alpha)
    #     return img

    # def saturation(self, img):
    #     """调整饱和度（仅前三个通道）"""
    #     rgb, data = self._split_data_channel(img)
    #     if random.randint(2):
    #         # 转换到HSV空间时确保使用前三个通道
    #         hsv = mmcv.bgr2hsv(rgb[..., ::-1].copy())  # RGB转BGR
    #         hsv[:, :, 1] = self.convert(
    #             hsv[:, :, 1],
    #             alpha=random.uniform(self.saturation_lower, self.saturation_upper)
    #         )[:, :, 0]  # 保持单通道
    #         rgb_processed = mmcv.hsv2bgr(hsv)[..., ::-1]
    #         return self._merge_data_channel(rgb_processed, data)
    #     return img

    # def hue(self, img):
    #     """调整色调（仅前三个通道）"""
    #     rgb, data = self._split_data_channel(img)
    #     if random.randint(2):
    #         hsv = mmcv.bgr2hsv(rgb[..., ::-1].copy())  # RGB转BGR
    #         hsv[:, :, 0] = (hsv[:, :, 0].astype(int) + 
    #                        random.randint(-self.hue_delta, self.hue_delta)) % 180
    #         rgb_processed = mmcv.hsv2bgr(hsv)[..., ::-1]
    #         return self._merge_data_channel(rgb_processed, data)
    #     return img
    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)#saturation has something wrong for 4 channels
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img) #it makes saturation to 0
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img
    def __call__(self, results):
        """处理流程（保证第四通道绝对不变）"""
        for img_key in ['img', 'B_img']:
            if img_key not in results:
                continue
                
            img = results[img_key]
            original_dtype = img.dtype
            max_num = img.max()
            min_num = img.min()
            if min_num < 0:
                img = img - min_num
            else:
                img = img.astype(np.float32)/max_num *255
            # modification
            rgb, data = self._split_data_channel(img)
            
            # 随机亮度
            rgb = self.brightness(rgb)[..., :3]
            
            # 随机对比度模式
            mode = random.randint(2)
            if mode == 1:
                rgb = self.contrast(rgb)[..., :3]
                
            # 随机饱和度
            rgb = self.saturation(rgb)[..., :3]
            
            # 随机色调
            rgb = self.hue(rgb)[..., :3]
            
            # 随机对比度
            if mode == 0:
                rgb = self.contrast(rgb)[..., :3]
            
            # 合并并恢复原始数据类型
            processed = self._merge_data_channel(rgb, data)
            if min_num < 0:
                processed = processed + min_num
            else:
                processed = processed.astype(np.float32)/255 *max_num
            results[img_key] = processed.astype(original_dtype)
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str

@PIPELINES.register_module()
class CopyImg(object):
    """Copy the image to avoid distortion
    Args:
    """
    def __init__(self):
        self.name="copy_image"

    def __call__(self, results):
        """

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images.
        """

        img = results['img']

        results['img_copy'] = img.copy()#May not have this name
        results['img_fields'].append('img_copy')
        return results

@PIPELINES.register_module()
class StrongAugmentation(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta

    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        # random brightness
        img = self.brightness(img)

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            img = self.contrast(img)

        # random saturation
        img = self.saturation(img)

        # random hue
        img = self.hue(img)

        # random contrast
        if mode == 0:
            img = self.contrast(img)

        results['img_strong_aug'] = img
        results['img_fields'].append('img_strong_aug')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@PIPELINES.register_module()
class StrongAugmentation4C_DataChannel(object):

    def __init__(self,
                 brightness_delta=32,
                 contrast_range=(0.5, 1.5),
                 saturation_range=(0.5, 1.5),
                 hue_delta=18):
        self.brightness_delta = brightness_delta
        self.contrast_lower, self.contrast_upper = contrast_range
        self.saturation_lower, self.saturation_upper = saturation_range
        self.hue_delta = hue_delta
    def _split_data_channel(self, img):
        if img.shape[-1] == 4:
            rgb, data = img[..., :3], img[..., 3:]
            return rgb, data
        return img, None
    def _merge_data_channel(self, rgb, data):
        if data is not None:
            return np.concatenate([rgb, data], axis=-1)
        return rgb
    def convert(self, img, alpha=1, beta=0):
        """Multiple with alpha and add beat with clip."""
        img = img.astype(np.float32) * alpha + beta
        #img = np.clip(img, 0, 255)
        return img#.astype(np.uint8)

    def brightness(self, img):
        """Brightness distortion."""
        if random.randint(2):
            return self.convert(
                img,
                beta=random.uniform(-self.brightness_delta,
                                    self.brightness_delta))
        return img

    def contrast(self, img):
        """Contrast distortion."""
        if random.randint(2):
            return self.convert(
                img,
                alpha=random.uniform(self.contrast_lower, self.contrast_upper))
        return img

    def saturation(self, img):
        """Saturation distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)
            img[:, :, 1] = self.convert(
                img[:, :, 1],
                alpha=random.uniform(self.saturation_lower,
                                     self.saturation_upper))
            img = mmcv.hsv2bgr(img)
        return img

    def hue(self, img):
        """Hue distortion."""
        if random.randint(2):
            img = mmcv.bgr2hsv(img)#has something strange
            img[:, :,
                0] = (img[:, :, 0].astype(int) +
                      random.randint(-self.hue_delta, self.hue_delta)) % 180
            img = mmcv.hsv2bgr(img)
        return img

    def __call__(self, results):
        """Call function to perform photometric distortion on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        img = results['img']
        max_num = img.max()
        min_num = img.min()#for sar data, minimum number is smaller than 0
        if min_num < 0:
            img = img - min_num
        else:
            img = img.astype(np.float32)/max_num *255
        # random brightness
        rgb, data = self._split_data_channel(img) #modification
        rgb = self.brightness(rgb) #confirm if the rgb2hsv support float32

        # mode == 0 --> do random contrast first
        # mode == 1 --> do random contrast last
        mode = random.randint(2)
        if mode == 1:
            rgb = self.contrast(rgb)

        # random saturation
        rgb = self.saturation(rgb)#no effect?

        # random hue
        rgb = self.hue(rgb)

        # random contrast
        if mode == 0:
            rgb = self.contrast(rgb)
        processed = self._merge_data_channel(rgb, data)
        if min_num < 0:
            processed = processed + min_num#is it neccsary to add minimum?
        else:
            img = img.astype(np.float32)/255 * max_num
        #convert the augmentated data
        results['img_strong_aug'] = processed
        results['img_fields'].append('img_strong_aug')
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(brightness_delta={self.brightness_delta}, '
                     f'contrast_range=({self.contrast_lower}, '
                     f'{self.contrast_upper}), '
                     f'saturation_range=({self.saturation_lower}, '
                     f'{self.saturation_upper}), '
                     f'hue_delta={self.hue_delta})')
        return repr_str


@PIPELINES.register_module()
class ClipNormalize(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True, axis=None, to_uint8=False):
        self.mean = np.array(mean).astype(np.float32)
        self.std = np.array(std).astype(np.float32)
        self.to_rgb = to_rgb
        self.to_uint8 = to_uint8

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        mean = self.mean.reshape(1, 1, -1)
        std = self.std.reshape(1, 1, -1)
        min_value = mean - 2 * std
        max_value = mean + 2 * std

        for img_key in results['img_fields']:
            img = results[img_key]
            img = (img.astype(np.float32) - min_value) / (max_value - min_value)
            img = np.clip(img, 0, 1)

            if self.to_rgb:
                img = img[:, :, [2,1,0]]

            if self.to_uint8:
                img = (img * 255).astype(np.uint8)

            results[img_key] = img

        return results

@PIPELINES.register_module()
class Uint82Float(object):
    def __call__(self, results):
        for img_key in results['img_fields']:
            results[img_key] = results[img_key].astype(np.float32) / 255
        return results

