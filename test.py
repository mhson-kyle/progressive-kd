import os
import json
import logging
import argparse
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from datasets.dataset import CTPDataset, ToTensor
from models.unet import UNet
from utils.config import Config
from utils.utils import *

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_dir', type=str, default='CTP_experiment', help='Experiment name')
    parser.add_argument('--ckpt', type=str,  default='best', help='Checkpoint to load')
    parser.add_argument('--stride_xy', type=int, default=16, help='stride to slide the patchwise inference of whole image')
    parser.add_argument('--stride_z', type=int, default=4, help='stride to slide the patchwise inference of whole image')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    if not os.path.exists(args.experiment_dir):
        raise FileNotFoundError(f'experiment does not exist: {os.path.basename(args.experiment_dir)}')
    config = Config(f'{args.experiment_dir}/config.yaml')

    test_result_path = os.path.join(args.experiment_dir, f"test_result_{args.ckpt}_{datetime.now().strftime('%y%m%d_%H%M%S')}")
    test_result_output_path = os.path.join(test_result_path, 'output')
    os.makedirs(test_result_path, exist_ok=True)
    os.makedirs(test_result_output_path, exist_ok=True)

    # Dataset
    logging.info(f"Loading test dataset from {config.cfg.get('data_dir')}")
    test_dataset = CTPDataset(data_dir=config.cfg.get('data_dir'),
                              split='test',  
                              transform=transforms.Compose([
                                  ToTensor()
                                  ]), 
                              reduced_rate=config.cfg.get('reduced_rate'), 
                              internal=config.cfg.get('internal'))
    testloader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    print(f'test dataset size: {len(test_dataset)}')
    
    # Model
    logging.info(f'Loading model from {args.experiment_dir}')
    model = UNet(in_channels=config.cfg.get('patch_z'), 
                 n_filters=config.cfg.get('num_filters'), 
                 normalization='batchnorm', 
                 branches=3).cuda()
    checkpoint = os.path.join(args.experiment_dir, 'ckpt', f'{args.ckpt}.h5')
    print(f'init model ED weight from {checkpoint}')
    model.load_state_dict(torch.load(checkpoint)['model_state_dict'], strict=False)
    model.eval()

    # Initialize output metrics
    output_arr_dict = {}
    output_metric_dict = {}
    aggregate_output_dict = {}
    for map_type in ['tmax', 'cbv', 'cbf']:
        aggregate_output_dict[map_type] = {}
        for metric in ['rmse', 'mae', 'scc', 'pcc', 'ssim', 'psnr']:
            aggregate_output_dict[map_type][metric] = []

    for _, batch in tqdm(enumerate(testloader)):
        test_result = test_single_case(model=model,
                                       batch=batch,
                                       stride_xy=args.stride_xy,
                                       stride_z=args.stride_z,
                                       patch_size=(config.cfg.get('patch_x'),
                                                   config.cfg.get('patch_y'),
                                                   config.cfg.get('patch_z')))

        casename = os.path.basename(test_result['casename'])
        slab_name = casename[:-14]
        slice_idx = int(casename[-8:-5])

        if slab_name not in output_arr_dict.keys():
            output_arr_dict[slab_name] = {}
            output_arr_dict[slab_name]['pred_tmax'] = {}
            output_arr_dict[slab_name]['pred_cbv'] = {}
            output_arr_dict[slab_name]['pred_cbf'] = {}
            
            output_arr_dict[slab_name]['gt_tmax'] = {}
            output_arr_dict[slab_name]['gt_cbv'] = {}
            output_arr_dict[slab_name]['gt_cbf'] = {}
            output_arr_dict[slab_name]['mask'] = {}


        output_arr_dict[slab_name]['pred_tmax'][slice_idx] = test_result['pred_tmax']
        output_arr_dict[slab_name]['pred_cbv'][slice_idx] = test_result['pred_cbv']
        output_arr_dict[slab_name]['pred_cbf'][slice_idx] = test_result['pred_cbf']

        output_arr_dict[slab_name]['gt_tmax'][slice_idx] = test_result['gt_tmax']
        output_arr_dict[slab_name]['gt_cbv'][slice_idx] = test_result['gt_cbv']
        output_arr_dict[slab_name]['gt_cbf'][slice_idx] = test_result['gt_cbf']

        output_arr_dict[slab_name]['mask'][slice_idx] = test_result['mask'].astype(np.uint8)

    # Save Result
    for slab_name, v in output_arr_dict.items():
        slab_output_path = os.path.join(test_result_output_path, slab_name)
        os.makedirs(slab_output_path, exist_ok=True)

        for map_type, outputs in v.items():
            map_out_path = os.path.join(slab_output_path, map_type)
            os.makedirs(map_out_path, exist_ok=True)

            for slice_idx, out_arr in outputs.items():
                img_path = os.path.join(map_out_path, '{}.png'.format(str(slice_idx).zfill(3)))
                save_01_img(img_path, out_arr)

    # Calculate Metric
    for slab_name, v in output_arr_dict.items():
        volumes_dict = {}

        for map_type, outputs in v.items():
            map_vol = []

            for slice_idx in sorted(list(outputs.keys())):
                map_vol.append(outputs[slice_idx])

            map_vol = np.stack(map_vol, -1)
            assert np.max(map_vol) <= 1.0
            assert np.min(map_vol) >= 0.0

            if 'tmax' in map_type:
                map_vol = map_vol * 24 #seconds
            elif 'cbv' in map_type:
                map_vol = map_vol * 200 #ml/100g
            elif 'cbf' in map_type:
                map_vol = map_vol * 1000 #ml/100g/min
            else:
                if map_type != 'mask':
                    raise NotImplementedError

            volumes_dict[map_type] = map_vol

        mask_3d = volumes_dict['mask']
        tmax_metric_result_dict = map_eval(volumes_dict['gt_tmax'], volumes_dict['pred_tmax'], 'tmax', mask_3d, aggregate_output_dict)
        cbv_metric_result_dict = map_eval(volumes_dict['gt_cbv'], volumes_dict['pred_cbv'], 'cbv', mask_3d, aggregate_output_dict)
        cbf_metric_result_dict = map_eval(volumes_dict['gt_cbf'], volumes_dict['pred_cbf'], 'cbf', mask_3d, aggregate_output_dict)


        output_metric_dict[slab_name] = {}
        output_metric_dict[slab_name]['shape'] = mask_3d.shape
        output_metric_dict[slab_name]['tmax'] = tmax_metric_result_dict
        output_metric_dict[slab_name]['cbv'] = cbv_metric_result_dict
        output_metric_dict[slab_name]['cbf'] = cbf_metric_result_dict

    for map_type, per_map_metric_output in aggregate_output_dict.items():
        for metric, metric_val_list in per_map_metric_output.items():
            aggregate_output_dict[map_type][metric] = np.mean(metric_val_list)

    with open(os.path.join(test_result_path, 'metric_result.txt'),'w') as f:
        json.dump(output_metric_dict, f, indent=4, cls=NumpyFloatValuesEncoder)
    with open(os.path.join(test_result_path, 'aggregate_result.txt'),'w') as f:
        json.dump(aggregate_output_dict, f, indent=4, cls=NumpyFloatValuesEncoder)

    print(aggregate_output_dict)
        
if __name__ == '__main__':
    main()