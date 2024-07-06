import os
import yaml  
from shutil import copyfile

from .utils import create_dir, set_logger


class Config:
    def __init__(self, config_path: str):
        self.config_path = config_path
        self.load_config()
        self.validate_args()
        self.setup_paths()
        self.compute_additional_params()

    def load_config(self):
        with open(self.config_path, 'r') as file:
            self.cfg = yaml.safe_load(file)
    
    def validate_args(self):
        # Validate root_path
        root_path = self.cfg.get('data_dir', '')
        if not os.path.exists(root_path):
            raise FileNotFoundError(f'Root path does not exist: {root_path}')
        
        # Validate teacher paths
        teacher_paths = self.cfg.get('teacher_experiment_names', [])
        for teacher_path in teacher_paths:
            if not os.path.exists(teacher_path):
                raise FileNotFoundError(f'Teacher experiment path does not exist: {teacher_path}')

    def setup_paths(self):
        experiment_name = self.cfg.get('student_experiment_name', 'CTP_student')
        self.experiment_output_path, self.snapshot_path, self.sample_output_path, self.log_path = create_dir(experiment_name)
        set_logger(self.cfg, self.log_path)
        copyfile(self.config_path, os.path.join(self.experiment_output_path, 'config.yaml'))

    def compute_additional_params(self):
        self.patch_size = (
            self.cfg.get('patch_x', 128),
            self.cfg.get('patch_y', 128),
            self.cfg.get('patch_z', 32)
        )
        self.gpu_batch_size = self.cfg.get('batch_size', 64)
        self.decoder_branches = self.cfg.get('decoder_branches', 3)
        self.input_channels = self.cfg.get('patch_z', 32)
