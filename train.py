import os
import argparse
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torchmetrics import StructuralSimilarityIndexMeasure
from tqdm import tqdm

from utils.config import Config
from utils.utils import seed
from models import ModelManager
from datasets import DatasetManager


class Trainer:
    def __init__(self, model_manager: ModelManager, dataset_manager: DatasetManager, config: Config, device: torch.device):
        self.model_manager = model_manager
        self.dataset_manager = dataset_manager
        self.config = config
        self.device = device
        self.writer = SummaryWriter(os.path.join(self.config.experiment_output_path, 'log'))

        self.student = self.model_manager.get_student()
        self.teachers = self.model_manager.get_teachers()

        self.train_loader = self.dataset_manager.get_train_loader()
        self.val_loader = self.dataset_manager.get_val_loader()

        # Optimizer
        self.optimizer = optim.Adam(
            self.student.parameters(),
            lr=self.config.cfg.get('lr_ed', 1e-2),
            betas=tuple(self.config.cfg.get('betas', [0.5, 0.99]))
        )

        if self.config.cfg.get('resume_from', 'None') != 'None':
            self.load_checkpoint()

        # Loss Functions
        self.mse_loss = nn.MSELoss().to(self.device)
        self.mae_loss = nn.L1Loss().to(self.device)
        self.ssim_loss = StructuralSimilarityIndexMeasure(data_range=1.0).to(self.device)
        self.kd_loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.map_loss_compute = self.create_map_loss_compute()

        # Training
        self.lambda_student = self.config.cfg.get('lambda_student', 0.5)
        self.T = self.config.cfg.get('T', 5)
        self.max_epoch = self.config.cfg.get('max_epoch', 100)

        # Teacher
        self.teacher_names = list(self.teachers.keys())
        self.teacher_idx = 0
        self.patience = self.config.cfg.get('patience', 5)
        self.current_patience = 0
        self.best_val_loss = float('inf')

    def create_map_loss_compute(self):
        def compute(pred_map, gt_map):
            ssim = 1 - self.ssim_loss(pred_map, gt_map)
            mae = self.mae_loss(pred_map, gt_map)
            return ssim + mae
        return compute

    def adjust_learning_rate(self, epoch):
        lr = self.config.cfg.get('lr_ed', 1e-2)
        decay_rate = self.config.cfg.get('lr_decay_rate', 0.1)
        decay_epochs = self.config.cfg.get('lr_decay_epochs', 30)
        lr = lr * (decay_rate ** (epoch // decay_epochs))
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr
    
    def process_data_teacher(self, data_student, subset_idx, reduced_rate):
        if reduced_rate != 1:
            indices = np.arange(subset_idx, data_student.shape[1], reduced_rate)
            data = data_student[:, indices, :, :]
            data = torch.repeat_interleave(data, reduced_rate, dim=1)
            if data.shape[1] < 40:
                pad_length = 40 - data.shape[1]
                channel = data[:, -1, :, :].unsqueeze(1)  # shape [batch, channels, 1]
                repeated_channels = channel.repeat(1, pad_length, 1, 1)
                data = torch.cat((data, repeated_channels), dim=1)
            data = data[:, :32, :, :]
        else:
            data = data_student
        return data

    def train(self):
        logging.info('Start Training')
        for epoch in tqdm(range(self.config.cfg.get('start_epoch', 0), self.max_epoch), desc='Training Epochs'):
            self.student.train()
            epoch_losses = {
                'tmax_ssim': 0.0, 'cbv_ssim': 0.0, 'cbf_ssim': 0.0,
                'tmax_mae': 0.0, 'cbv_mae': 0.0, 'cbf_mae': 0.0,
                'tmax_kd': 0.0, 'cbv_kd': 0.0, 'cbf_kd': 0.0,
                'features_kd': 0.0
            }

            lr = self.adjust_learning_rate(epoch)
            self.writer.add_scalar('lr/lr_ED', lr, epoch)

            for iter_num, batch in enumerate(self.train_loader):
                data_student = batch['data_student'].to(self.device)
                tmax = batch['tmax'].to(self.device)
                cbv = batch['cbv'].to(self.device)
                cbf = batch['cbf'].to(self.device)
                casename = batch['casename']

                # Student forward pass
                out_student, _, bottleneck_student = self.student(data_student)
                out_map_tmax_student = out_student['tmax']
                out_map_cbv_student = out_student['cbv']
                out_map_cbf_student = out_student['cbf']

                # Supervised Loss
                loss_sl = (
                    self.map_loss_compute(out_map_tmax_student, tmax) +
                    self.map_loss_compute(out_map_cbv_student, cbv) +
                    self.map_loss_compute(out_map_cbf_student, cbf)
                ) * 0.2 + (
                    self.mae_loss(out_map_tmax_student, tmax) +
                    self.mae_loss(out_map_cbv_student, cbv) +
                    self.mae_loss(out_map_cbf_student, cbf)
                ) * 0.7

                # Knowledge Distillation Loss
                if self.teachers and self.teacher_idx < len(self.teacher_names):
                    data_teacher = batch['data_teacher'].to(self.device)
                    current_teacher_name = self.teacher_names[self.teacher_idx]
                    current_teacher = self.teachers[current_teacher_name]
                    data_teacher = self.process_data_teacher(data_teacher, int(casename[0].split('_')[-1]), int(current_teacher_name.split('_')[-1]))
                    with torch.no_grad():
                        out_teacher, _, bottleneck_teacher = current_teacher(data_teacher)

                    out_map_tmax_kd = out_teacher['tmax']
                    out_map_cbv_kd = out_teacher['cbv']
                    out_map_cbf_kd = out_teacher['cbf']

                    tmax_kd_loss = self.kd_loss_fn(
                        F.log_softmax(out_map_tmax_student / self.T, dim=1),
                        F.softmax(out_map_tmax_kd / self.T, dim=1)
                    )
                    cbv_kd_loss = self.kd_loss_fn(
                        F.log_softmax(out_map_cbv_student / self.T, dim=1),
                        F.softmax(out_map_cbv_kd / self.T, dim=1)
                    )
                    cbf_kd_loss = self.kd_loss_fn(
                        F.log_softmax(out_map_cbf_student / self.T, dim=1),
                        F.softmax(out_map_cbf_kd / self.T, dim=1)
                    )
                    features_kd_loss = self.mse_loss(
                        bottleneck_student,
                        bottleneck_teacher
                    )
                    loss_kd = tmax_kd_loss + cbv_kd_loss + cbf_kd_loss + features_kd_loss
                    loss = (1 - self.lambda_student) * loss_sl + self.lambda_student * (self.T ** 2) * loss_kd
                else:
                    loss = loss_sl

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Logging
                self.writer.add_scalar('loss/total_loss', loss.item(), epoch * len(self.train_loader) + iter_num)
                self.writer.add_scalar('loss/loss_sl', loss_sl.item(), epoch * len(self.train_loader) + iter_num)
                logging.info(f'Epoch {epoch+1}/{self.max_epoch} | Batch {iter_num+1}/{len(self.train_loader)} | Loss: {loss.item():.4f}| Loss SL: {loss_sl.item():.4f} ')
                if self.teachers and self.teacher_idx < len(self.teachers):
                    self.writer.add_scalar('loss/loss_kd', loss_kd.item(), epoch * len(self.train_loader) + iter_num)

                # Accumulate epoch losses
                epoch_losses['tmax_ssim'] += (1 - self.ssim_loss(out_map_tmax_student, tmax)).item()
                epoch_losses['cbv_ssim'] += (1 - self.ssim_loss(out_map_cbv_student, cbv)).item()
                epoch_losses['cbf_ssim'] += (1 - self.ssim_loss(out_map_cbf_student, cbf)).item()

                epoch_losses['tmax_mae'] += self.mae_loss(out_map_tmax_student, tmax).item()
                epoch_losses['cbv_mae'] += self.mae_loss(out_map_cbv_student, cbv).item()
                epoch_losses['cbf_mae'] += self.mae_loss(out_map_cbf_student, cbf).item()

                # If using teacher, accumulate KD losses
                if self.teachers and self.teacher_idx < len(self.teachers):
                    epoch_losses['tmax_kd'] += tmax_kd_loss.item()
                    epoch_losses['cbv_kd'] += cbv_kd_loss.item()
                    epoch_losses['cbf_kd'] += cbf_kd_loss.item()
                    epoch_losses['features_kd'] += features_kd_loss.item()

            avg_train_losses = {k: v / iter_num for k, v in epoch_losses.items()}
            

            # Validation
            avg_val_loss = self.validate(epoch)
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                self.current_patience = 0
                self.save_checkpoint(epoch, 'best')
                logging.info(f'Validation loss improved to {self.best_val_loss:.4f}.')
            else:
                self.current_patience += 1
                logging.info(f'No improvement in validation loss. Current patience: {self.current_patience}/{self.patience}')

            if epoch % self.config.cfg.get('save_interval', 10) == 0:
                self.save_checkpoint(epoch, 'latest')
                
            if self.teachers and self.teacher_idx < len(self.teachers) - 1:
                if self.current_patience >= self.patience:
                    self.teacher_idx += 1
                    self.current_patience = 0
                    logging.info(f'Switching to teacher {self.teacher_idx + 1}/{len(self.teachers)} due to patience exceeded.')
            elif self.teachers and self.teacher_idx == len(self.teachers) - 1:
                if self.current_patience >= self.patience:
                    logging.info('Patience exceeded, but no more teachers to switch to.')

            for loss_name, avg_loss in avg_train_losses.items():
                self.writer.add_scalar(f'train/{loss_name}', avg_loss, epoch + 1)

    def validate(self, epoch):
        self.student.eval()
        val_loss = 0.0

        with torch.no_grad():
            for iter_num, batch in enumerate(self.val_loader):
                data_student = batch['data_student'].to(self.device)
                tmax = batch['tmax'].to(self.device)
                cbv = batch['cbv'].to(self.device)
                cbf = batch['cbf'].to(self.device)
                casename = batch['casename']

                out_student, _, bottleneck_student = self.student(data_student)
                out_map_tmax_student = out_student['tmax']
                out_map_cbv_student = out_student['cbv']
                out_map_cbf_student = out_student['cbf']

                # Supervised Loss
                loss_sl = (
                    self.map_loss_compute(out_map_tmax_student, tmax) +
                    self.map_loss_compute(out_map_cbv_student, cbv) +
                    self.map_loss_compute(out_map_cbf_student, cbf)
                ) * 0.2 + (
                    self.mae_loss(out_map_tmax_student, tmax) +
                    self.mae_loss(out_map_cbv_student, cbv) +
                    self.mae_loss(out_map_cbf_student, cbf)
                ) * 0.7

                # Knowledge Distillation Loss
                if self.teachers and self.teacher_idx < len(self.teacher_names):
                    data_teacher = batch['data_teacher'].to(self.device)
                    current_teacher_name = self.teacher_names[self.teacher_idx]
                    current_teacher = self.teachers[current_teacher_name]
                    data_teacher = self.process_data_teacher(data_teacher, int(casename[0].split('_')[-1]), int(current_teacher_name.split('_')[-1]))

                    with torch.no_grad():
                        out_teacher, _, bottleneck_teacher = current_teacher(data_teacher)

                    out_map_tmax_kd = out_teacher['tmax']
                    out_map_cbv_kd = out_teacher['cbv']
                    out_map_cbf_kd = out_teacher['cbf']

                    tmax_kd_loss = self.kd_loss_fn(
                        F.log_softmax(out_map_tmax_student / self.T, dim=1),
                        F.softmax(out_map_tmax_kd / self.T, dim=1)
                    )
                    cbv_kd_loss = self.kd_loss_fn(
                        F.log_softmax(out_map_cbv_student / self.T, dim=1),
                        F.softmax(out_map_cbv_kd / self.T, dim=1)
                    )
                    cbf_kd_loss = self.kd_loss_fn(
                        F.log_softmax(out_map_cbf_student / self.T, dim=1),
                        F.softmax(out_map_cbf_kd / self.T, dim=1)
                    )
                    features_kd_loss = self.mse_loss(
                        bottleneck_student,
                        bottleneck_teacher
                    )
                    loss_kd = tmax_kd_loss + cbv_kd_loss + cbf_kd_loss + features_kd_loss
                    loss = (1 - self.lambda_student) * loss_sl + self.lambda_student * (self.T ** 2) * loss_kd
                else:
                    loss = loss_sl

                val_loss += loss.item()

        avg_val_loss = val_loss / iter_num
        self.writer.add_scalar('val/loss', avg_val_loss, epoch + 1)
        logging.info(f'Validation Loss: {avg_val_loss:.4f}')
        return avg_val_loss

    def save_checkpoint(self, epoch, filename='latest'):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.student.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        save_path = os.path.join(self.config.snapshot_path, f'{filename}.h5')
        torch.save(checkpoint, save_path)
        logging.info(f'Saved checkpoint: {save_path}')
    
    def load_checkpoint(self):
        checkpoint_path = os.path.join(self.config.cfg.get('resume_from'), f'ckpt_epoch_{self.config.cfg.get('start_epoch', 0)}.pth')
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f'Checkpoint not found at {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.student.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.iteration = checkpoint.get('iteration', 0)
        logging.info(f'Model Loaded: {checkpoint_path} at iteration {self.iteration}')

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train Student Model with Knowledge Distillation')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()
    return args

def main():
    args = parse_arguments()
    config = Config(args.config)
    seed(config.cfg.get('seed', 1234))
    device = torch.device(f'cuda' if torch.cuda.is_available() else 'cpu')
    
    dataset_manager = DatasetManager(config)
    model_manager = ModelManager(config, device)
    trainer = Trainer(model_manager, dataset_manager, config, device)
    trainer.train()

if __name__ == '__main__':
    main()