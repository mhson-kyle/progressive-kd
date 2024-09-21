import logging

from torch.utils.data import DataLoader
from torchvision import transforms

from .dataset import CTPDataset, RandomFlip, RandomCrop, ToTensor


class DatasetManager:
    def __init__(self, config):
        self.config = config
        self.train_loader, self.val_loader = self.setup_dataloaders()

    def setup_dataloaders(self):
        logging.info('Setting up DataLoaders')

        train_dataset = CTPDataset(
            data_dir=self.config.cfg.get('data_dir', '/data/kyle/CTP/DATA/'),
            split='train',
            transform=transforms.Compose([
                RandomFlip(self.config.cfg.get('flip_prob', 0.5)),
                RandomCrop(self.config.patch_size, self.config.cfg.get('crop_offset', 5)),
                ToTensor(),
                ]),
            internal=self.config.cfg.get('internal'),
            reduced_rate=self.config.cfg.get('reduced_rate', 1)
        )
        print(f'Train Dataset size: {len(train_dataset)}')

        val_dataset = CTPDataset(
            data_dir=self.config.cfg.get('data_dir', '/data/kyle/CTP/DATA/'),
            split='val',
            transform=transforms.Compose([
                RandomCrop(self.config.patch_size, self.config.cfg.get('crop_offset', 5)),
                ToTensor()]),
            internal=self.config.cfg.get('internal'),
            reduced_rate=self.config.cfg.get('reduced_rate', 1)
        )
        print(f'Validation Dataset size: {len(val_dataset)}')

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.gpu_batch_size,
            shuffle=True,
            num_workers=self.config.cfg.get('num_workers', 16),
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.gpu_batch_size,
            shuffle=False,
            num_workers=self.config.cfg.get('num_workers', 16),
            pin_memory=True
        )
        return train_loader, val_loader

    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader
