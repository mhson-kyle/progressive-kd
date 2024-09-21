import os
import random
from glob import glob

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class CTPDataset(Dataset):
    def __init__(self, data_dir=None, split='train', transform=None, reduced_rate=1, tmax_clip=24, internal=False):
        self.transform = transform
        self.data_list = []
        self.split = split
        self.reduced_rate = reduced_rate
        self.tmax_clip = tmax_clip
        if internal:
            if self.split == 'train':
                isles_file = sorted(
                    glob(os.path.join(data_dir, 'by_slice/isles/imagesTr', '*.h5')))
                isles_file += sorted(glob(os.path.join(data_dir,
                                     'by_slice/isles/imagesTs', '*.h5')))
                stanford_file = sorted(
                    glob(os.path.join(data_dir, 'by_slice/stanford/imagesTr', '*.h5')))
                isles_file = isles_file[:int(len(isles_file) * 0.9)]
                stanford_file = stanford_file[:int(len(stanford_file) * 0.9)]
                self.data_list = isles_file + stanford_file

            elif self.split == 'val':
                isles_file = sorted(glob(os.path.join(data_dir, 'by_slice/isles/imagesTr', '*.h5')))
                isles_file += sorted(glob(os.path.join(data_dir,'by_slice/isles/imagesTs', '*.h5')))
                stanford_file = sorted(glob(os.path.join(data_dir, 'by_slice/stanford/imagesTr', '*.h5')))
                isles_file = isles_file[int(len(isles_file) * 0.9):]
                stanford_file = stanford_file[int(len(stanford_file) * 0.9):]
                self.data_list = isles_file + stanford_file

            elif self.split == 'test':
                self.data_list = sorted(glob(os.path.join(data_dir, 'by_slice/stanford/imagesTs', '*.h5')))
        else:
            if self.split == 'train':
                isles_file = sorted(glob(os.path.join(data_dir, 'by_slice/isles/imagesTr', '*.h5')))
                stanford_file = sorted(glob(os.path.join(data_dir, 'by_slice/stanford/imagesTr', '*.h5')))
                stanford_file += sorted(glob(os.path.join(data_dir,'by_slice/stanford/imagesTs', '*.h5')))
                isles_file = isles_file[:int(len(isles_file) * 0.9)]
                stanford_file = stanford_file[:int(len(stanford_file) * 0.9)]
                self.data_list = isles_file + stanford_file

            elif self.split == 'val':
                isles_file = sorted(glob(os.path.join(data_dir, 'by_slice/isles/imagesTr', '*.h5')))
                stanford_file = sorted(glob(os.path.join(data_dir, 'by_slice/stanford/imagesTr', '*.h5')))
                stanford_file += sorted(glob(os.path.join(data_dir,'by_slice/stanford/imagesTs', '*.h5')))
                isles_file = isles_file[int(len(isles_file) * 0.9):]
                stanford_file = stanford_file[int(len(stanford_file) * 0.9):]
                self.data_list = isles_file + stanford_file

            elif self.split == 'test':
                self.data_list = sorted(glob(os.path.join(data_dir, 'by_slice/isles/imagesTs', '*.h5')))

    def __len__(self):
        if self.split == 'train':
            return len(self.data_list) * self.reduced_rate
        return len(self.data_list)

    def __getitem__(self, idx):
        if (self.reduced_rate != 1) & (self.split == 'train'):
            base_idx = idx // self.reduced_rate
            subset_idx = idx % self.reduced_rate
            image_path = self.data_list[base_idx]
            casename = os.path.basename(image_path) + '_' + str(subset_idx)
        else:
            image_path = self.data_list[idx]
            casename = os.path.basename(image_path) + '_0'

        h5f = h5py.File(image_path, 'r')
        data_teacher = np.array(h5f['slices_over_time'], dtype=np.float32)
        start_end = h5f['start_end']
        data_student = data_teacher[..., start_end[0]: start_end[1]]
        assert data_student.shape[-1] == 40, f'Data shape is {data_student.shape}'

        # Temopral Resolution
        if self.reduced_rate != 1:
            if self.split == 'test' or self.split == 'val':
                subset_idx = 0
            indices = np.arange(
                subset_idx, data_student.shape[-1], self.reduced_rate)

            data_student = data_student[..., indices]
            data_student = np.repeat(data_student, repeats=self.reduced_rate, axis=-1)
            if data_student.shape[-1] < 40:
                _ = 40 - data_student.shape[-1]
                last_channel = data_student[:, :, -1:]
                repeated_channels = np.repeat(last_channel, _, axis=2)
                data_student = np.concatenate((data_student, repeated_channels), axis=2)
            data_student = data_student[:, :, :40]

        assert data_student.shape[-1] == 40, f'Data shape is {data_student.shape}'
        tmax = np.array(h5f['tmax'], dtype=np.float32)
        cbf = np.array(h5f['cbf'], dtype=np.float32)
        cbv = np.array(h5f['cbv'], dtype=np.float32)

        if 'isles' in casename:
            # ISLES2018
            tmax = np.clip(tmax, 0, self.tmax_clip) / (self.tmax_clip)
            cbf = np.clip(cbf, 0, 1000) / 1000
            cbv = np. clip(cbv, 0, 200) / 200
        elif 'stanford_' in casename:
            # Stanford 1st Batch
            tmax = np.clip(tmax, 0, (self.tmax_clip * 10)) / (self.tmax_clip * 10)
            cbf = np.clip(cbf, 0, 1000) / 1000
            cbv = np.clip(cbv, 0, 200) / 200
        elif 'stanford2nd_' in casename:
            # Stanford 2nd Batch
            tmax = np.clip(tmax, 0, (self.tmax_clip * 100)) / (self.tmax_clip * 100)
            cbf = np.clip(cbf, 0, 10000) / 10000
            cbv = np.clip(cbv, 0, 2000) / 2000
        elif 'stanfordperfbad_' in casename:
            # Stanford 3rd Batch
            tmax = np.clip(tmax, 0, (self.tmax_clip * 100)) / (self.tmax_clip * 100)
            cbf = np.clip(cbf, 0, 10000) / 10000
            cbv = np.clip(cbv, 0, 2000) / 2000

        sample = {
            'data_student': data_student,
            'data_teacher': data_teacher,
            'tmax': tmax,
            'cbv': cbv,
            'cbf': cbf,
            'casename': casename,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


class RandomCrop(object):
    """
    Randomly crop the image in a sample

    Args:
        output_size (int): Desired output size
        offset_magnitude_ratio (float): Offset magnitude ratio
    """

    def __init__(self, output_size, offset_magnitude_ratio):
        self.output_size = output_size
        self.offset_magnitude_ratio = offset_magnitude_ratio

    def cut(self, data_student, data_teacher, tmax=None, cbv=None, cbf=None, sizes=None):

        w1, h1, d1 = sizes

        data_student = data_student[h1:h1 + self.output_size[1], w1:w1 + self.output_size[0], d1:d1 + self.output_size[2]]
        data_teacher = data_teacher[h1:h1 + self.output_size[1], w1:w1 + self.output_size[0], d1:d1 + self.output_size[2]]
        tmax = tmax[h1:h1 + self.output_size[1], w1:w1 + self.output_size[0]]
        cbv = cbv[h1:h1 + self.output_size[1], w1:w1 + self.output_size[0]]
        cbf = cbf[h1:h1 + self.output_size[1], w1:w1 + self.output_size[0]]

        return data_student, data_teacher, tmax, cbv, cbf

    def __call__(self, sample):
        data_student, data_teacher, tmax, cbv, cbf = sample['data_student'], sample['data_teacher'], sample['tmax'], sample['cbv'], sample['cbf']

        (w, h, d) = data_student.shape

        while True:
            w1 = np.random.randint(0, w - self.output_size[0])
            h1 = np.random.randint(0, h - self.output_size[1])

            d1 = 0 if d <= self.output_size[2] else np.random.randint(0, d - self.output_size[2])

            data_student_cut, data_teacher_cut, tmax_cut, cbv_cut, cbf_cut = self.cut(
                data_student, data_teacher, tmax, cbv, cbf, (w1, h1, d1))

            if np.any(data_student_cut):
                break

        new_sample = sample.copy()
        new_sample['data_student'] = data_student_cut
        new_sample['data_teacher'] = data_teacher_cut
        new_sample['tmax'] = tmax_cut
        new_sample['cbv'] = cbv_cut
        new_sample['cbf'] = cbf_cut

        return new_sample


class RandomFlip(object):
    """
    Randomly flip the dataset in a sample

    Args:
        flip_p (float): Probability of flipping the data
    """

    def __init__(self, flip_p):
        self.flip_p = flip_p

    def __call__(self, sample):
        if random.random() <= self.flip_p:
            data_student, data_teacher, tmax, cbv, cbf = sample['data_student'], sample['data_teacher'], sample['tmax'], sample['cbv'], sample['cbf']

            new_sample = sample.copy()

            data_student = np.flip(data_student, axis=1)
            data_teacher = np.flip(data_teacher, axis=1)
            tmax = np.flip(tmax, axis=1)
            cbv = np.flip(cbv, axis=1)
            cbf = np.flip(cbf, axis=1)

            new_sample['data_student'] = data_student
            new_sample['data_teacher'] = data_teacher
            new_sample['tmax'] = tmax
            new_sample['cbv'] = cbv
            new_sample['cbf'] = cbf

            return new_sample
        else:
            return sample


class ToTensor(object):
    """
    Converts a data arrays to tensors and normalizes specific fields.
    """

    def __call__(self, sample):
        data_student, data_teacher, tmax, cbv, cbf = sample['data_student'], sample['data_teacher'], sample['tmax'], sample['cbv'], sample['cbf']

        data_student = data_student.transpose((2, 0, 1)).copy()
        data_teacher = data_teacher.transpose((2, 0, 1)).copy()
        tmax = tmax[np.newaxis, :, :].copy()
        cbv = cbv[np.newaxis, :, :].copy()
        cbf = cbf[np.newaxis, :, :].copy()

        data_student = data_student / 80.0  # rescale input to 0~1
        data_student = (data_student - 0.135) / 0.25  # normalize with mean&std

        data_teacher = data_teacher / 80.0  # rescale input to 0~1
        data_teacher = (data_teacher - 0.135) / 0.25  # normalize with mean&std

        new_sample = sample.copy()
        new_sample['data_student'] = torch.from_numpy(data_student)
        new_sample['data_teacher'] = torch.from_numpy(data_teacher)
        new_sample['tmax'] = torch.from_numpy(tmax)
        new_sample['cbv'] = torch.from_numpy(cbv)
        new_sample['cbf'] = torch.from_numpy(cbf)

        return new_sample
