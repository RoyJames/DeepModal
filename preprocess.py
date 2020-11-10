from analysis import frequency

import numpy as np
import os
from tqdm import tqdm
from glob import glob
import scipy
from torch_geometric.data import Data
import torch

freq = frequency.FrequencyScale(32)

def preprocess(dirname, data_dir):
    face_centers = np.load(dirname+'/face_centers.npy')
    face_normals = np.load(dirname+'/face_normals.npy')
    displacements = np.load(dirname+'/displacements.npy')
    poles_coeffs = np.load(dirname+'/poles_coeffs.npy')
    omegas = np.load(dirname+'/omegas.npy')
    
    p_dis = np.zeros((freq.resolution, displacements.shape[1]),dtype=np.float32)
    p_w = np.zeros((freq.resolution,poles_coeffs.shape[1]), dtype=np.complex64)
    p_s = np.zeros(freq.resolution, dtype=np.float32)
    modes_dict = {i:[] for i in range(freq.resolution)}
    for i,omega in enumerate(omegas):
        modes_dict[freq.omega2index(omega)].append(i)

    for i in range(freq.resolution):
        idxs = modes_dict[i]
        if len(idxs) == 0:
            continue
        displace = (displacements[idxs]**2).sum(0)**0.5
        coeff = poles_coeffs[idxs].mean(0)
        p_dis[i] = displace
        p_w[i] = coeff
        p_s[i] = 1

    p_dis = torch.FloatTensor(p_dis.T)
    p_w = torch.cat((torch.FloatTensor(p_w.real),torch.FloatTensor(p_w.imag)),-1).unsqueeze(0)
    p_s = torch.IntTensor(p_s).unsqueeze(0)
    pos = torch.FloatTensor(face_centers)
    normal = torch.FloatTensor(face_normals)
    x_in = torch.cat((pos,normal),-1)
    
    data = Data()
    data.x = x_in
    data.pos = pos
    data.normal = normal
    data.dis = p_dis
    data.w = p_w
    data.s = p_s

    scale = (1 / data.pos.abs().max()) * 0.999999
    data.pos = data.pos * scale
    torch.save(data, data_dir + os.path.basename(dirname) + '.pt')


    # print(data.x.shape)
    # print(data.pos.shape)
    # print(data.normal.shape)
    # print(data.dis.shape)
    # print(data.w.shape)
    # print(data.s.shape)
    # print(V,U)
    # print((np.abs(V)**2).sum())
    # print((np.abs(U)**2).sum())
    # print((p_dis[:,:32]**2).sum(0)+(p_dis[:,32:]**2).sum(0))



file_list_train = glob('dataset/*/train/*/displacements.npy')
file_list_test = glob('dataset/*/test/*/displacements.npy')


file_list_dic = {
    'test':file_list_test,
    'train':file_list_train
}
for n in ['train','test']:
    out_dir = '/home/jxt/ssd_dataset/' + n + '/'
    os.makedirs(out_dir, exist_ok=True)
    lst = file_list_dic[n]
    print(n,len(lst))
    for filename in tqdm(lst):
        preprocess(os.path.dirname(filename),out_dir)
    #     break
    # break
    

            
