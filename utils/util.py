import numpy as np
import torch
import random
import logging
import os
from datetime import datetime
import json
from skimage.io import imsave
from shutil import copyfile
import yaml
from munch import munchify

def load_config(config_file):
    with open(config_file) as file:
        config = yaml.safe_load(file)
    config = munchify(config)
    return config

def to255(data):
    # maxval = 1.0
    # minval = 0.0
    # return (((data - minval) / (maxval - minval))*255).astype(np.uint8)
    return (data*255.0).round().astype(np.uint8)

def to255_t(data):  # from tensor to numpy
    data = np.array(data)
    # maxval = 1.0
    # minval = 0.0
    # return (((data - minval) / (maxval - minval))*255).astype(np.uint8)
    return (data*255.0).round().astype(np.uint8)

def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def linear_rampup(current, rampup_length):
    """Linear rampup"""
    assert current >= 0 and rampup_length >= 0
    if current >= rampup_length:
        return 1.0
    else:
        return current / rampup_length

def cosine_rampdown(current, rampdown_length):
    """Cosine rampdown from https://arxiv.org/abs/1608.03983"""
    assert 0 <= current <= rampdown_length
    return float(.5 * (np.cos(np.pi * current / rampdown_length) + 1))

def get_current_consistency_weight(epoch, totalepoch, rampmax = 1.0):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    assert rampmax <= 1.0

    end_epoch = int(totalepoch * rampmax)
    if epoch > end_epoch:
        return 1.0
    else:
        return 1.0 * sigmoid_rampup(epoch, end_epoch)

# def get_domain_num(name_batch):

#     names_num = []

#     for n in name_batch:
#         if 'stanford' in n:
#             names_num.append([1])
#         else:
#             names_num.append([0])

#     return torch.from_numpy(np.array(names_num)).float().cuda()

def get_domain_num(domain_batch):

    domains_num = []

    for isPublic in domain_batch:
        domains_num.append([isPublic])

    return torch.from_numpy(np.array(domains_num)).long().cuda().squeeze()

def seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
def create_dir(experiment_name):
    try:
        experiment_output_path = experiment_name + '_{}'.format(datetime.now().strftime('%y%m%d_%H%M%S'))
        print(f'Experiment Name: {experiment_output_path}')
        experiment_output_path = os.path.join('Experiments', experiment_output_path)
        snapshot_path = os.path.join(experiment_output_path, 'ckpt')
        sample_output_path = os.path.join(experiment_output_path, 'sample_train_output')
        log_path = os.path.join(experiment_output_path, 'log')
        
        for path in [experiment_output_path, snapshot_path, sample_output_path, log_path]:
            os.makedirs(path, exist_ok=True)
        
        return experiment_output_path, snapshot_path, sample_output_path
    
    except Exception as e:
        logging.error(f"Error creating directories: {e}")

def create_test_dir(experiment_name):
    experiment_output_path = os.path.join('Experiments', experiment_name)

    if not os.path.exists(experiment_output_path):
        raise FileNotFoundError('experiment does not exist: {}'.format(os.path.basename(experiment_output_path)))
    snapshot_path = os.path.join(experiment_output_path,'ckpt')

    test_result_save_path = os.path.join(experiment_output_path, 'test_result_{}'.format(datetime.now().strftime('%y%m%d_%H%M%S')))
    test_output_save_path = os.path.join(experiment_output_path, 'test_result_{}'.format(datetime.now().strftime('%y%m%d_%H%M%S')), 'output')
    os.makedirs(test_result_save_path, exist_ok=True)
    os.makedirs(test_output_save_path, exist_ok=True)

    return test_result_save_path, test_output_save_path, snapshot_path

def set_logger(args, log_dir):
    logging_level = logging.getLevelName(args.loglevel.upper())
    logging.basicConfig(
        level=logging_level,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        filename=os.path.join(log_dir, 'training_log.txt'),
        filemode='w',
        encoding='utf-8'
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging_level)
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    console_handler.setFormatter(formatter)
    logging.getLogger('').addHandler(console_handler)
    def serialize(obj):
        """Custom serialization for objects not supported natively by JSON."""
        if isinstance(obj, torch.device):
            return str(obj)  # Convert device object to string
        raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")

    with open(os.path.join(log_dir, 'options.txt'), 'w') as f:
        json.dump(args.__dict__, f, default=serialize, indent=2)
    
def worker_init_fn(worker_id):
    # https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    
def save_map_montage(sampled_batch, out_map_student, sample_output_path, iteration):
    """
    Processes the batch data and saves a montage image of the maps.

    :param sampled_batch: The batch of sampled data.
    :param out_map_student: The output maps from the model.
    :param iteration: The current iteration number.
    :param sample_output_path: The path where the montage image will be saved.
    :param to255_t: Function to convert tensor to 255 scale image.
    """
    tmax = to255_t(sampled_batch['tmax'][0, 0, :, :])
    cbv = to255_t(sampled_batch['cbv'][0, 0, :, :])
    cbf = to255_t(sampled_batch['cbf'][0, 0, :, :])

    out_map_tmax = to255_t(out_map_student['out_map_tmax'][0, 0, :, :].detach().cpu())
    out_map_cbv = to255_t(out_map_student['out_map_cbv'][0, 0, :, :].detach().cpu())
    out_map_cbf = to255_t(out_map_student['out_map_cbf'][0, 0, :, :].detach().cpu())

    map_montage = np.vstack([
        np.hstack([tmax, out_map_tmax]),
        np.hstack([cbv, out_map_cbv]),
        np.hstack([cbf, out_map_cbf])
    ])

    imsave(f"{sample_output_path}/{iteration}_map_0.png", map_montage, cmap='jet')

def save_checkpoint(model, optimizer, snapshot_path, epoch_num, save_name=None):
        """
        Saves the model checkpoint.
        :param model: The model to be saved.
        :param snapshot_path: Base path for saving the model.
        :param epoch_num: The current epoch number.
        """
        if save_name is not None:
            save_path = os.path.join(snapshot_path, f'iter_{save_name}.h5')
        else:
            save_path = os.path.join(snapshot_path, f'iter_{epoch_num}.h5')
            
        logging.info(f"Saving model to {snapshot_path} at iter {epoch_num}")
        torch.save({
            'model_state_dict' : model.module.state_dict() if torch.distributed.is_initialized() else model.state_dict(), 
            'optimizer_state_dict' : optimizer.state_dict(),
            'epoch' : epoch_num,
        }, save_path)

def load_checkpoint(model, optimizer, checkpoint):
    model_dict = model.state_dict()
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return model, optimizer, checkpoint['epoch']
    
def log_metrics(epoch_num, loss_maps, da_loss=None):
    logging.info(
        f"EPOCH {epoch_num} |  "
        f"TMAX {loss_maps['tmax'].item():.3f} "
        f"CBV {loss_maps['cbv'].item():.3f} CBF {loss_maps['cbf'].item():.3f} "
    )
    
def copy_source(file, output_dir):
    copyfile(file, os.path.join(output_dir, os.path.basename(file)))



def get_output_dir(dir, filename):
    t = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    output_dir = os.path.join(dir, 'output/' + filename, t)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir