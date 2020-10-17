from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch_geometric.transforms import RadiusGraph, NormalizeScale
from net.graph_net import UNet
from net.dataset import ModalDataset
from torch_geometric.data import DataLoader
from torchvision import transforms
import argparse
import numpy as np
import sys

def start():

    def forward(loader):
        total_loss = []
        if not args.debug:
            loader = tqdm(loader)

        for batch in loader:
            data = batch['data'].to(device)
            out = model(data).view(data.y.shape)
            loss = F.l1_loss(out, data.y)
            if torch.is_grad_enabled():
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if args.debug:
                check_grad()

            if np.isnan(loss.item()):
                # torch.save(data, './error_data.pt')
                # torch.save(out, './error_out.pt')
                print('!!!!!!!!!!!!!!!!!NaN output!!!!!!!!!!!!!!!!')
                return

            total_loss.append(loss.item())

        return np.mean(total_loss)

    def check_grad():
        print('=========================GRAD CHECK===========================')
        for name,param in model.named_parameters():
            print(name)
            print(f'value:{param.data}')
            print(f'gradient:{param.grad}') 

    def get_loader(phase, bz = 16):
        return DataLoader(ModalDataset(
                args.dataset + phase, 
                pre_transform = pre_transform,
                ),batch_size=bz)

    #==========================initialize================================
    device = torch.device(f'cuda:{args.cuda}')
    model = UNet(3,64,96).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    pre_transform = transforms.Compose([RadiusGraph(0.21/32), NormalizeScale()]) 
    loader = {
        'train': get_loader('train'),
        'test' : get_loader('test'),
        'valid' : get_loader('valid')
    }
    torch.set_grad_enabled(False)
    #==========================train=====================================
    for epoch in range(20):
        model.train()
        print(f'============={epoch}=============')
        with torch.set_grad_enabled(True):
            print('train loss:{:.5f}'.format(forward(loader['train'])))
        model.eval()
        print('test loss:{:.5f}'.format(forward(loader['test'])))
        print('valid loss:{:.5f}'.format(forward(loader['valid'])))

    #==========================test=======================================
    loader = get_loader('test', 1)
    i = 0
    for batch in loader:
        data = batch['data'].to(device)
        out = model(data).view(data.y.shape)
        np.save(batch['path'][0] + '/vecs_norm_.npy',out.cpu().numpy())
        i += 1
        if i > 50:
            break



if __name__ == "__main__":
    #==========================args parser=============================
    parser = argparse.ArgumentParser(description='Train GNN to estimate modal synthesis')
    parser.add_argument('--cuda', type=int, default = 0, help='Cuda index')
    parser.add_argument('--dataset', type=str, default='G:/dataset/', help='Dataset root directory')
    parser.add_argument('--debug', dest='debug', action='store_true')
    args = parser.parse_args()
    args.debug = args.debug

    if args.debug:
        sys.stdout = open("output.txt", "w")
    torch.set_printoptions(linewidth=160)
    start()
    if args.debug:
        sys.stdout.close()
    
    





    



