import os
from tqdm import tqdm
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


def get_data(configs):
    files = os.listdir(configs.path_root)
    files = sorted(files)
    sample_data = []
    for file in tqdm(files):
        real_path = os.path.join(configs.path_root, file)
        image = Image.open(real_path)
        image = image.convert('L')
        image = image.resize((configs.img_width, configs.img_width))
        data = np.expand_dims(np.array(image), -1).astype(np.float32)
        sample_data.append(data)
    sample_data = np.stack(sample_data, 0)
    return sample_data


class SplitDataset(Dataset):
    def __init__(self, data, configs):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.configs = configs

    def __getitem__(self, item):
        # (b, 64, 64, 1)
        inputs = self.data[item: item + self.configs.total_length]
        mask_true = torch.zeros((self.configs.pred_length - 1, self.configs.img_width, self.configs.img_width, 1),
                                dtype=torch.float32)
        return inputs, mask_true

    def __len__(self):
        return len(self.data) - self.configs.total_length