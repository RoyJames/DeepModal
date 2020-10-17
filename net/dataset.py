import os.path as osp
import torch
from torch.utils.data import Dataset
import os
from glob import glob
from torch_geometric.data import Data
import numpy as np
from tqdm import tqdm

class ModalDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__()
        self.transform = transform
        self.pre_transform = pre_transform
        self.root = root
        self.data_path_list = glob(f'{self.root}/*')

        #========cut half=========
        self.data_path_list = self.data_path_list[:len(self.data_path_list)//2]
        #=========================

        self.pre_process()
        self.processed_file_names = glob(f'{self.processed_dir}/*')
        
    def pre_process(self):
        self.processed_dir = self.root.replace('G','E')
        if os.path.exists(self.processed_dir):
            return
        os.mkdir(self.processed_dir)
        print('pre process')
        for i, data_path in  enumerate(tqdm(self.data_path_list)):
            x = torch.FloatTensor(np.load(data_path+'/vertices.npy'))
            y = torch.FloatTensor(np.load(data_path+'/vecs_norm.npy'))
            y = y.reshape(y.shape[0]//3, 3, -1)
            data = Data(x=x, y=y, pos=x)
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            torch.save(data, osp.join(self.processed_dir, 'data_{}.pt'.format(i)))

    # def check(self):
    #     skip = len(self.data_path_list) // 2
    #     for i, data_path in  enumerate(tqdm(self.data_path_list)):
    #         if i < skip:
    #             continue
    #         x = torch.FloatTensor(np.load(data_path+'/vertices.npy'))
    #         y = torch.FloatTensor(np.load(data_path+'/vecs_norm.npy'))
    #         if torch.isnan(x).any() or torch.isnan(y).any():
    #             print(data_path)
            
    def __len__(self):
        return len(self.processed_file_names)

    def __getitem__(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        if self.transform is not None:
            data = self.transform(data)
        return {'data':data, 'path':self.data_path_list[idx]}