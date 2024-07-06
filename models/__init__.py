import logging
import os

import torch

from .unet import UNet
from .vnet import VNet


class ModelManager:
    def __init__(self, config, device: torch.device):
        self.config = config
        self.device = device

        self.student_model = self.create_student_model()
        self.teacher_models = self.create_teacher_models()

    def create_student_model(self):
        logging.info('Creating Student Model')
        if self.config.cfg.get('model') == 'UNet':
            model = UNet(
                in_channels=self.config.input_channels,
                n_filters=self.config.cfg.get('num_filters', 32),
                normalization='batchnorm',
                branches=self.config.decoder_branches
            ).to(self.device)
            model.initialize_weights()

        elif self.config.cfg.get('model') == 'VNet':
            model = VNet(
                in_channels=self.config.input_channels,
                n_filters=self.config.cfg.get('num_filters', 32),
                branches=self.config.decoder_branches
            ).to(self.device)
            model.initialize_weights()
        else:
            raise ValueError(f"Model {self.config.cfg.get('model')} not supported")

        return model

    def create_teacher_models(self):
        logging.info('Creating Teacher Models')
        teacher_models = {}
        teacher_weights = self.config.cfg.get('teacher_ckpt', [])

        for teacher_weight in teacher_weights:
            teacher_model = UNet(
                n_channels=32,  # Assuming teacher has 32 input channels
                n_filters=32,
                normalization='batchnorm',
                branches=self.config.decoder_branches
            ).to(self.device)
            teacher_model.load_state_dict(torch.load(teacher_weight, map_location=self.device)['model_state_dict'], strict=False)
            teacher_model.eval()
            teacher_model.train(False)
            teacher_models[os.path.basename(teacher_weight).split('.')[0]] = teacher_model
            logging.info(f'Loaded Teacher Model: {teacher_weight}')

        return teacher_models

    def get_student(self):
        return self.student_model

    def get_teachers(self):
        return self.teacher_models
