from tqdm import tqdm
import torch
import torch.nn.functional as F
from net.graph_net import Feature_extractor, dgcnn_classification, dgcnn_segmentation
from net.dataset import ModalDataset, RandomSample
from torch_geometric.data import DataLoader
from torch_scatter import scatter
import argparse
import numpy as np
import sys

def check_grad(model):
    print('=========================GRAD CHECK===========================')
    for name,param in model.named_parameters():
        print(name)
        print(f'value:{param.data}')
        print(f'gradient:{param.grad}')

def start():
    def normalize(x, batch):
        sum_value = scatter(x**2, batch, dim=0, reduce='sum')
        sum_value = sum_value[batch]
        out = x/(sum_value + 1e-20)
        return out

    def forward(loader):
        total_loss = []
        for data in tqdm(loader):
            data = data.to(device)
            out = model(data)
            out = normalize(out, data.batch)
            mask = scatter(data.dis**2, data.batch, dim=0, reduce='sum')
            out = out * mask[data.batch]
            loss = F.l1_loss(out, data.dis)
            if torch.is_grad_enabled():
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss.append(loss.item())
        return np.mean(total_loss)

    def get_loader(phase, bz = 10):
        return DataLoader(ModalDataset(args.dataset + phase, transform = RandomSample()),batch_size=bz,num_workers=10)

    #==========================initialize================================
    device = torch.device(f'cuda:{args.cuda}')
    model = dgcnn_segmentation(32).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    loader = {
        'train': get_loader('train'),
        'test' : get_loader('test'),
        'valid' : get_loader('valid')
    }
    torch.set_grad_enabled(False)
    #==========================train=====================================
    best_loss = 1e10
    for epoch in range(50):
        model.train()
        print(f'============={epoch}=============')
        with torch.set_grad_enabled(True):
            print('train loss:{:.5f}'.format(forward(loader['train'])))
        model.eval()
        print('test loss:{:.5f}'.format(forward(loader['test'])))
        valid_loss = forward(loader['valid'])
        print('valid loss:{:.5f}'.format(valid_loss))
        if valid_loss < best_loss:
            best_loss = valid_loss
            torch.save(model.state_dict(), 'weights/dis_weights.pt')
        scheduler.step()
    
    #==========================test=======================================
    # loader = get_loader('test', 1)
    # i = 0
    # for batch in loader:
    #     data = batch['data'].to(device)
    #     out = model(data).view(data.y.shape)
    #     np.save(batch['path'][0] + '/vecs_norm_.npy',out.cpu().numpy())
    #     i += 1
    #     if i > 50:
    #         break



if __name__ == "__main__":
    #==========================args parser=============================
    parser = argparse.ArgumentParser(description='Train GNN to estimate modal synthesis')
    parser.add_argument('--cuda', type=int, default = 0, help='Cuda index')
    parser.add_argument('--dataset', type=str, default='/home/jxt/ssd_dataset/', help='Dataset root directory')
    parser.add_argument('--debug', dest='debug', action='store_true')
    args = parser.parse_args()
    start()


    # if args.debug:
    #     sys.stdout = open("output.txt", "w")
    # torch.set_printoptions(linewidth=160)
    
    # if args.debug:
    #     sys.stdout.close()
    
    





    



