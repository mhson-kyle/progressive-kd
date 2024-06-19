# Copyright (c) ByteDance, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import sys
from datetime import datetime

from networks import dist

def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser for the training script.')

    # Environment
    parser.add_argument('--experiment_name', type=str, default='CTP_Pretrain', help='Experiment name')
    parser.add_argument('--data_path', type=str, default='/localdata/mhson/urop/new_data/', help='Data path')
    parser.add_argument('--init_weight', type=str, default='', help='Initial model weight')
    parser.add_argument('--checkpoint', type=str, default='', help='Resume from checkpoint')
    parser.add_argument('--loglevel', type=str, default='info', help='Logging level')
    parser.add_argument('--gpu', type=str, default='0', help='Logging level')
    parser.add_argument('--device', type=str, default='cuda:0', help='Logging level')
    parser.add_argument('--seed', type=int, default=1234, help='Logging level')

    # Encoder hyperparameters
    parser.add_argument('--mask', type=float, default=0.6, help='Mask ratio')
    parser.add_argument('--model', type=str, default='your_convnet_small', help='Model type')
    parser.add_argument('--input_size', type=int, default=128, help='Input size')
    parser.add_argument('--sbn', type=bool, default=True, help='Use sbn')
    
    # Data hyperparameters
    parser.add_argument('--bs', type=int, default=512, help='Batch size')
    parser.add_argument('--dataloader_workers', type=int, default=8, help='Number of dataloader workers')
    
    # pre-training hyperparameters
    parser.add_argument('--dropout_rate', type=float, default=0.0, help='Dropout')
    parser.add_argument('--base_lr', type=float, default=2e-4, help='Base learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.04, help='Weight decay')
    parser.add_argument('--weight_decay_end', type=float, default=0.2, help='Weight decay end')
    parser.add_argument('--epochs', type=int, default=1600, help='Number of epochs')
    parser.add_argument('--warmup_epochs', type=int, default=40, help='Number of warmup epochs')
    parser.add_argument('--clip', type=int, default=5., help='Gradient clipping')
    parser.add_argument('--opt', type=str, default='lamb', help='Optimizer')
    parser.add_argument('--ada', type=float, default=0., help='Ada')

    # Parse and return arguments
    return parser.parse_args()
 
def get_args(exp_dir=None):
    from utils import misc

    # Initialize
    args = parse_args()

    misc.init_distributed_environ(exp_dir=exp_dir)
    

    # Update args
    if not dist.initialized():
        args.sbn = False
    args.first_logging = True
    args.device = dist.get_device()
    args.batch_size_per_gpu = args.bs // dist.get_world_size()
    args.glb_batch_size = args.batch_size_per_gpu * dist.get_world_size()

    args.ada = args.ada or 0.95
    args.densify_norm = 'bn'

    args.opt = args.opt.lower()
    args.lr = args.base_lr * args.glb_batch_size / 256
    args.weight_decay_end = args.weight_decay_end or args.weight_decay

    return args

if __name__ == "__main__":
    args = get_args()
    print(f'initial args:\n{str(args)}')
