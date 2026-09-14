# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp
import shutil
import time
import warnings
import sys
sys.path.append("./")
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import mmcv
import torch
from mmcv.cnn.utils import revert_sync_batchnorm
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint,
                         wrap_fp16_model)
from mmcv.utils import DictAction

from rsiseg import digit_version
from rsiseg.apis import multi_gpu_test, single_gpu_test
from rsiseg.datasets import build_dataloader, build_dataset
from rsiseg.models import build_segmentor, build_test_model
from rsiseg.utils import setup_multi_processes
from copy import deepcopy
from thop import profile
from fvcore.nn import FlopCountAnalysis, parameter_count

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')
    parser.add_argument('--config', help='test config file path')
    parser.add_argument('--checkpoint', help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help=('if specified, the evaluation metric results will be dumped'
              'into the directory as json'))
    parser.add_argument(
        '--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument(
        '--format-only',
        action='store_true',
        help='Format the output results without perform evaluation. It is'
        'useful when you want to format the result to a specific format and '
        'submit it to the test server')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
        ' for generic datasets, and "cityscapes" for Cityscapes')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument(
        '--show-dir', help='directory where painted images will be saved')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='id of gpu to use '
        '(only applicable to non-distributed testing)')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu_collect is not specified')
    parser.add_argument(
        '--samples-per-gpu',
        type=int,
        default=1,
        help='test batch size')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        help="--options is deprecated in favor of --cfg_options' and it will "
        'not be supported in version v0.22.0. Override some settings in the '
        'used config, the key-value pair in xxx=yyy format will be merged '
        'into config file. If the value to be overwritten is a list, it '
        'should be like key="[a,b]" or key=a,b It also allows nested '
        'list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation '
        'marks are necessary and that no white space is allowed.')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        help='custom options for evaluation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--opacity',
        type=float,
        default=0.5,
        help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument(
        '--img_norm_type',
        type=str,
        default='img_norm_cfg',
        help='how to normalize the shown images.')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--revise_checkpoint_key', type=bool, default=False)

    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.cfg_options:
        raise ValueError(
            '--options and --cfg-options cannot be both '
            'specified, --options is deprecated in favor of --cfg-options. '
            '--options will not be supported in version v0.22.0.')
    if args.options:
        warnings.warn('--options is deprecated in favor of --cfg-options. '
                      '--options will not be supported in version v0.22.0.')
        args.cfg_options = args.options

    return args


def main():
    args = parse_args()
    assert args.out or args.eval or args.format_only or args.show \
        or args.show_dir, \
        ('Please specify at least one operation (save/eval/format/show the '
         'results / save the results) with the argument "--out", "--eval"'
         ', "--format-only", "--show" or "--show-dir"')

    if args.eval and args.format_only:
        raise ValueError('--eval and --format_only cannot be both specified')

    if args.out is not None and not args.out.endswith(('.pkl', '.pickle')):
        raise ValueError('The output file must be a pkl file.')

    cfg = mmcv.Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if args.aug_test:
        # hard code index
        cfg.data.test.pipeline[1].img_ratios = [
            0.5, 0.75, 1.0, 1.25, 1.5, 1.75
        ]
        cfg.data.test.pipeline[1].flip = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.gpu_id is not None:
        cfg.gpu_ids = [args.gpu_id]

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        cfg.gpu_ids = [args.gpu_id]
        distributed = False
        if len(cfg.gpu_ids) > 1:
            warnings.warn(f'The gpu-ids is reset from {cfg.gpu_ids} to '
                          f'{cfg.gpu_ids[0:1]} to avoid potential error in '
                          'non-distribute testing time.')
            cfg.gpu_ids = cfg.gpu_ids[0:1]
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    rank, _ = get_dist_info()
    # allows not to create
    if args.work_dir is not None and rank == 0:
        mmcv.mkdir_or_exist(osp.abspath(args.work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(args.work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(args.work_dir,
                                 f'eval_single_scale_{timestamp}.json')
    elif rank == 0:
        work_dir = osp.join('./work_dirs',
                            osp.splitext(osp.basename(args.config))[0])
        mmcv.mkdir_or_exist(osp.abspath(work_dir))
        timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
        if args.aug_test:
            json_file = osp.join(work_dir,
                                 f'eval_multi_scale_{timestamp}.json')
        else:
            json_file = osp.join(work_dir,
                                 f'eval_single_scale_{timestamp}.json')

    # build the dataloader
    # TODO: support multiple images per gpu (only minor changes are needed)
    dataset = build_dataset(cfg.data.test)
    # The default loader config
    loader_cfg = dict(
        # cfg.gpus will be ignored if distributed
        num_gpus=len(cfg.gpu_ids),
        dist=distributed,
        shuffle=False)
    # The overall dataloader settings
    loader_cfg.update({
        k: v
        for k, v in cfg.data.items() if k not in [
            'train', 'val', 'test', 'train_dataloader', 'val_dataloader',
            'test_dataloader'
        ]
    })
    test_loader_cfg = {
        **loader_cfg,
        'samples_per_gpu': args.samples_per_gpu,
        'shuffle': False,  # Not shuffle by default
        **cfg.data.get('test_dataloader', {})
    }
    # build the dataloader
    data_loader = build_dataloader(dataset, **test_loader_cfg)

    # build the model and load checkpoint
    cfg.model.train_cfg = None

    print(cfg)
    uda_model=build_test_model(cfg,test_cfg=cfg.get('test_cfg'))

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(uda_model)

    if args.revise_checkpoint_key:#to fit the loadin of dual model
        checkpoint = load_checkpoint(
            uda_model,
            # model,
            args.checkpoint,
            map_location='cpu',
            #revise_keys=[(r'^module\.', ''), ('model.', '')]
            )
    else:
        checkpoint = load_checkpoint(uda_model, args.checkpoint, map_location='cpu')


    if 'CLASSES' in checkpoint.get('meta', {}):
        uda_model.CLASSES = checkpoint['meta']['CLASSES']
    else:
        print('"CLASSES" not found in meta, use dataset.CLASSES instead')
        uda_model.CLASSES = dataset.CLASSES
    if 'PALETTE' in checkpoint.get('meta', {}):
        uda_model.PALETTE = checkpoint['meta']['PALETTE']
    else:
        print('"PALETTE" not found in meta, use dataset.PALETTE instead')
        uda_model.PALETTE = dataset.PALETTE

    # clean gpu memory when starting a new evaluation.
    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options

    # Deprecated
    efficient_test = eval_kwargs.get('efficient_test', False)
    if efficient_test:
        warnings.warn(
            '``efficient_test=True`` does not have effect in tools/test.py, '
            'the evaluation and format results are CPU memory efficient by '
            'default')

    eval_on_format_results = (
        args.eval is not None and 'cityscapes' in args.eval)
    if eval_on_format_results:
        assert len(args.eval) == 1, 'eval on format results is not ' \
                                    'applicable for metrics other than ' \
                                    'cityscapes'

    if args.format_only or eval_on_format_results:
        if 'imgfile_prefix' in eval_kwargs:
            tmpdir = eval_kwargs['imgfile_prefix']
        else:
            tmpdir = '.format_cityscapes'
            eval_kwargs.setdefault('imgfile_prefix', tmpdir)
        mmcv.mkdir_or_exist(tmpdir)
    else:
        tmpdir = None

    if not distributed:
        warnings.warn(
            'SyncBN is only supported with DDP. To be compatible with DP, '
            'we convert SyncBN to BN. Please use dist_train.sh which can '
            'avoid this error.')
        if not torch.cuda.is_available():
            assert digit_version(mmcv.__version__) >= digit_version('1.4.4'), \
                'Please use MMCV >= 1.4.4 for CPU training!'
        uda_model = revert_sync_batchnorm(uda_model)

        profile_results = improved_profile_uda_model(uda_model)
        print(profile_results)
        uda_model = MMDataParallel(uda_model, device_ids=cfg.gpu_ids)
        results, results_img, results_comb = single_gpu_test(
            uda_model,#
            #model,
            data_loader,
            args.show,
            args.show_dir,
            False,
            #efficient_test=True,
            args.opacity,
            pre_eval=args.eval is not None and not eval_on_format_results,
            #pre_eval=True,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs,
            img_norm_type=args.img_norm_type
        )
    else:
        uda_model = MMDistributedDataParallel(
            uda_model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
        results = multi_gpu_test(
            uda_model,
            data_loader,
            args.tmpdir,
            args.gpu_collect,
            False,
            pre_eval=args.eval is not None and not eval_on_format_results,
            format_only=args.format_only or eval_on_format_results,
            format_args=eval_kwargs)

    rank, _ = get_dist_info()
    if rank == 0:
        if args.out:
            warnings.warn(
                'The behavior of ``args.out`` has been changed since MMSeg '
                'v0.16, the pickled outputs could be seg map as type of '
                'np.array, pre-eval results or file paths for '
                '``dataset.format_results()``.')
            print(f'\nwriting results to {args.out}')
            mmcv.dump(results, args.out)
        if args.eval:#evaluation process
            eval_kwargs.update(metric=args.eval)
            metric = dataset.evaluate(results, **eval_kwargs)
            metric_dict = dict(config=args.config, metric=metric)
            mmcv.dump(metric_dict, json_file, indent=4)
            metric_img = dataset.evaluate(results_img, **eval_kwargs)
            metric_dict_img = dict(config=args.config, metric=metric_img)
            mmcv.dump(metric_dict_img, json_file, indent=4)
            metric_comb = dataset.evaluate(results_comb, **eval_kwargs)
            metric_dict_comb = dict(config=args.config, metric=metric_comb)
            mmcv.dump(metric_dict_comb, json_file, indent=4)
            if tmpdir is not None and eval_on_format_results:
                # remove tmp dir when cityscapes evaluation
                shutil.rmtree(tmpdir)



def profile_uda_model(uda_model, input_shape=(4, 3, 256, 256), device="cuda:0"):
    results = {}
    submodels = ['model', 'ema_model', 'dual_model','Gs', 'Gt', 'Ds', 'Dt']
    
    for submodel_name in submodels:
        if hasattr(uda_model, submodel_name):
            submodel = getattr(uda_model, submodel_name)
            submodel = submodel.to(device)
            input_tensor = torch.randn(input_shape).to(device)
            gt_semantic_seg = torch.randint(0, 6, (4, 1, 256, 256)).to(device)
            try:
                flops = FlopCountAnalysis(submodel, (input_tensor,  gt_semantic_seg))
                total_flops = flops.total()
                total_params = parameter_count(submodel)['']
                results[submodel_name] = {
                    'flops': total_flops,
                    'params': total_params
                }
            except Exception as e:
                try:
                    batch_size = input_shape[0]
                    img_metas = [
                        {
                            'img_shape': (input_shape[2], input_shape[3], input_shape[1]),
                            'scale_factor': 1.0,
                            'flip': False,
                            'flip_direction': 'horizontal'
                        } for _ in range(batch_size)
                    ]
                    flops = FlopCountAnalysis(submodel, (input_tensor, img_metas))
                    total_flops = flops.total()
                    total_params = parameter_count(submodel)['']
                    results[submodel_name] = {
                        'flops': total_flops,
                        'params': total_params
                    }
                except Exception as e2:
                    try:
                        batch_size = input_shape[0]
                        img_metas = [
                            {
                                'img_shape': (input_shape[2], input_shape[3], input_shape[1]),
                                'scale_factor': 1.0,
                                'flip': False,
                                'flip_direction': 'horizontal'
                            } for _ in range(batch_size)
                        ]
                        flops = FlopCountAnalysis(submodel, (input_tensor, gt_semantic_seg))
                        total_flops = flops.total()
                        total_params = parameter_count(submodel)['']
                        results[submodel_name] = {
                            'flops': total_flops,
                            'params': total_params
                        }
                    except Exception as e3:
                        print(f"Failed to profile {submodel_name} with fvcore: {e2}")
                        results[submodel_name] = None
    return results
def improved_profile_uda_model(uda_model, device="cuda:0", input_shape=(4, 3, 256, 256)):
    """改进的UDA模型统计函数，有更好的错误处理和回退机制"""
    
    results = {}
    submodels = ['model', 'ema_model', 'dual_model', 'Gs', 'Gt', 'Ds', 'Dt']

    batch_size = input_shape[0]
    input_tensor = torch.randn(input_shape).to(device)
    gt_semantic_seg = torch.randint(0, 6, (batch_size, 1, 256, 256)).to(device)

    img_metas = [
        {
            'img_shape': (input_shape[2], input_shape[3], input_shape[1]),
            'scale_factor': 1.0,
            'flip': False,
            'flip_direction': 'horizontal'
        } for _ in range(batch_size)
    ]
    
    for submodel_name in submodels:
        if hasattr(uda_model, submodel_name):
            submodel = getattr(uda_model, submodel_name).to(device)
            print(f"Profiling {submodel_name}...")
            
            if submodel_name in ['model', 'ema_model', 'dual_model']:
                success = profile_complex_model(submodel, submodel_name, input_tensor, 
                                              img_metas, gt_semantic_seg, results)
            else:
                success = profile_simple_model(submodel, submodel_name, input_tensor, results)
            
            if not success:
                try:
                    total_params = parameter_count(submodel)['']
                    results[submodel_name] = {
                        'params': total_params,
                        'flops': 0,
                        'note': 'Params only, FLOPs failed'
                    }
                    print(f"{submodel_name:15} | {'N/A':>8} | {total_params/1e6:6.2f} M Params (params only)")
                except Exception as e:
                    print(f"完全失败 to profile {submodel_name}: {e}")
                    results[submodel_name] = None
    
    return results

def profile_complex_model(submodel, submodel_name, input_tensor, img_metas, gt_semantic_seg, results):
    """统计复杂训练模型"""
    attempts = [
        lambda: FlopCountAnalysis(submodel, (input_tensor, img_metas, input_tensor, gt_semantic_seg, 
                                           input_tensor, img_metas, input_tensor, input_tensor)),
        lambda: FlopCountAnalysis(submodel, (input_tensor, img_metas, gt_semantic_seg)),
        lambda: FlopCountAnalysis(submodel, (input_tensor, gt_semantic_seg)),
        lambda: FlopCountAnalysis(submodel, (input_tensor,)),
    ]
    
    for i, attempt_fn in enumerate(attempts):
        try:
            flops = attempt_fn()
            total_flops = flops.total()
            total_params = parameter_count(submodel)['']
            
            results[submodel_name] = {
                'flops': total_flops,
                'params': total_params,
                'attempt': i + 1
            }
            
            print(f"{submodel_name:15} | {total_flops/1e9:8.2f} GFLOPs | {total_params/1e6:6.2f} M Params (attempt {i+1})")
            return True
            
        except Exception as e:
            if i < len(attempts) - 1:
                print(f"  Attempt {i+1} failed: {e}, trying next...")
            else:
                print(f"  All attempts failed for {submodel_name}")
    
    return False

def profile_simple_model(submodel, submodel_name, input_tensor, results):
    attempts = [
        lambda: FlopCountAnalysis(submodel, (input_tensor,)),
        lambda: FlopCountAnalysis(submodel, (input_tensor, input_tensor)),
    ]
    
    for i, attempt_fn in enumerate(attempts):
        try:
            flops = attempt_fn()
            total_flops = flops.total()
            total_params = parameter_count(submodel)['']
            
            results[submodel_name] = {
                'flops': total_flops,
                'params': total_params,
                'attempt': i + 1
            }
            
            print(f"{submodel_name:15} | {total_flops/1e9:8.2f} GFLOPs | {total_params/1e6:6.2f} M Params (attempt {i+1})")
            return True
            
        except Exception as e:
            if i < len(attempts) - 1:
                print(f"  Attempt {i+1} failed: {e}, trying next...")
            else:
                print(f"  All attempts failed for {submodel_name}")
    
    return False
if __name__ == '__main__':
    main()
