
from net.graph_net import Feature_extractor, dgcnn_classification, dgcnn_segmentation
from net.dataset import ModalDataset, RandomSample
import torch
import torch.nn.functional as F
from torch_geometric.transforms import RandomRotate
from torch_geometric.data import DataLoader
from torch_scatter import scatter


import argparse
import numpy as np
import sys
from tqdm import tqdm

def start(data_type):
    def check_grad():
        print('=========================GRAD CHECK===========================')
        for name,param in model.named_parameters():
            print(name)
            print(f'value:{param.data}')
            print(f'gradient:{param.grad}') 

    def complex_matmul(A1,B1,A2,B2):
        real_part = torch.matmul(A1,A2)-torch.matmul(B1,B2)
        imag_part = torch.matmul(A1,B2)+torch.matmul(B1,A2)
        return [real_part, imag_part]

    def forward(loader):
        total_loss = []
        for data in tqdm(loader):
            data = data.to(device)
            if data_type == 'w':
                out = model(data).view(-1,32,200)
                out = out * data.s.unsqueeze(-1)
                out_real = out[...,:100].unsqueeze(-2)
                out_imag = out[...,100:].unsqueeze(-2)
                w_real = data.w[...,:100].unsqueeze(-2)
                w_imag = data.w[...,100:].unsqueeze(-2)
                out_ = complex_matmul(out_real, out_imag, pole_matrix_real, pole_matrix_imag)
                w_ = complex_matmul(w_real, w_imag, pole_matrix_real, pole_matrix_imag)
                loss = F.l1_loss(out_[0], w_[0]) + F.l1_loss(out_[1],w_[1])

            elif data_type == 'dis':
                out = model(data)
                out = out * data.s[data.batch]
                loss = F.l1_loss(out, data.dis)

            elif data_type == 's':
                out = model(data).reshape(-1,2)
                loss = F.cross_entropy(out, data.s.reshape(-1))

            if torch.is_grad_enabled():
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss.append(loss.item())
        return np.mean(total_loss)

    def get_loader(phase, bz = 10):
        transforms = [RandomSample(2048)] 
        if data_type == 'dis':
            transforms += [RandomRotate(180,0),RandomRotate(180,1),RandomRotate(180,2)]
        return DataLoader(ModalDataset(args.dataset + phase, transforms = transforms),batch_size=bz,num_workers=10)

    #==========================initialize================================
    device = torch.device(f'cuda:{args.cuda}')
    if data_type == 'w':
        model = dgcnn_classification(32*200).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
        pole_matrix = np.load('weights/pole_matrixs.npy')
        pole_matrix = np.transpose(pole_matrix, (0,2,1))
        pole_matrix_real = torch.FloatTensor(pole_matrix.real).to(device)
        pole_matrix_imag = torch.FloatTensor(pole_matrix.imag).to(device)
    elif data_type == 'dis':
        model = dgcnn_segmentation(32).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    elif data_type == 's':
        model = dgcnn_segmentation(32*2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    loader = {
        'train': get_loader('train'),
        'test' : get_loader('test')
    }
    torch.set_grad_enabled(False)
    #==========================train=====================================
    for epoch in range(200):
        model.train()
        print(f'============={epoch}=============')
        with torch.set_grad_enabled(True):
            print('train loss:{:.5f}'.format(forward(loader['train'])))
        model.eval()
        print('test loss:{:.5f}'.format(forward(loader['test'])))
        torch.save(model.state_dict(), f'weights/{data_type}_weights.pt')
        scheduler.step()


if __name__ == "__main__":
    #==========================args parser=============================
    parser = argparse.ArgumentParser(description='Train GNN to estimate modal synthesis')
    parser.add_argument('--cuda', type=int, default = 0, help='Cuda index')
    parser.add_argument('--dataset', type=str, default='/home/jxt/ssd_dataset/', help='Dataset root directory')
    parser.add_argument('--type', type=str,  help='dis, w or s')
    args = parser.parse_args()
    start(args.type)

    





    



