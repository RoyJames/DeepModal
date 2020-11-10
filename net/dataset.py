import os.path as osp
import torch
from torch.utils.data import Dataset
import os
from glob import glob
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm
import math
import random
import torch
from torch_geometric.transforms import LinearTransformation

class RandomRotate():

    def __init__(self, degrees = 180):
        self.degrees = degrees

    def __call__(self, data):
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        matrix1 = np.array([[1, 0, 0], [0, cos, sin], [0, -sin, cos]])
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        matrix2 = np.array([[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]])
        degree = math.pi * random.uniform(*self.degrees) / 180.0
        sin, cos = math.sin(degree), math.cos(degree)
        matrix3 = np.array([[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]])
        matrix = torch.tensor(matrix1.dot(matrix2).dot(matrix3))
        data.pos = torch.matmul(data.pos, matrix.to(data.pos.dtype))
        data.normal = torch.matmul(data.normal, matrix.to(data.normal.dtype))
        return data 

def RandomSample(n = 1024):
    def transform(data):
        faces_num = len(data.x)
        if faces_num < n:
            replacement = True
        else:
            replacement = False
        mask = torch.multinomial(torch.ones(faces_num), n, replacement=replacement)
        data.normal = data.normal[mask]
        data.dis = data.dis[mask]
        data.pos = data.pos[mask]
        data.x = data.x[mask]
        return data
    return transform


class ModalDataset(Dataset):
    def __init__(self, root, transforms=[]):
        super().__init__()
        self.data_list = glob(f'{root}/*')
        print(f'size of dataset {root} is {len(self.data_list)}')
        self.transforms = transforms

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = torch.load(self.data_list[idx])
        for transform in self.transforms:
            data = transform(data)
        data.w = data.w / 1e7
        return data