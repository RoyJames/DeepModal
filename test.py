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
from glob import glob
import random
import bempp.api
from bempp.api import GridFunction, export
from analysis.BEM import boundary_mesh, PolesMatrix, SPEED_OF_SOUND
import os
from torch_geometric.data import Data
from analysis import frequency

def test():
    device = torch.device(f'cuda:0')

    model_dis = dgcnn_segmentation(32).to(device)
    model_dis.load_state_dict(torch.load('weights/dis_weights.pt'))
    model_dis.eval()
    model_w = dgcnn_classification(32*200).to(device)
    model_w.load_state_dict(torch.load('weights/w_weights.pt'))
    model_w.eval()
    pole_matrix = PolesMatrix()
    sphere = boundary_mesh(grid=bempp.api.shapes.sphere(h=0.02,r = 0.2))
    freq = frequency.FrequencyScale(32)
    torch.set_grad_enabled(False)
    # test_file_list = glob('/home/jxt/ssd_dataset/test/bowl0080')
    # select_list = random.sample(test_file_list, 1)
    
    def work(filename):
        dirname = os.path.basename(filename)[:-3]
        output_dir = 'modedata/' + dirname
        os.makedirs(output_dir, exist_ok=True)
        dirname = glob('dataset/*/test/'+dirname)[0]

        print(dirname)
        vertices = np.load(dirname+'/vertices.npy')
        omegas = np.load(dirname+'/omegas.npy')
        poles_coeffs = np.load(dirname+'/poles_coeffs.npy')
        boundary_faces = np.load(dirname+'/boundary_faces.npy')
        mesh = boundary_mesh(vertices, boundary_faces)
        
        n = 2048
        data = torch.load(filename)
        faces_num = len(data.x)
        mask_list = torch.ones(faces_num)
        #==================================w====================
        if mask_list.sum() > n:
                mask = torch.multinomial(mask_list, n, replacement=False)
        else:
            mask = torch.multinomial(mask_list, n, replacement=True)

        data1 = Data()
        data1.pos = data.pos[mask]
        data1.normal = data.normal[mask]
        data1.batch = torch.zeros(len(data1.pos),dtype=torch.int64)
        data1 = data1.to(device)
        out = model_w(data1).view(32,200).cpu().numpy()
        out_w = out[:,:100] + 1j*out[:,100:]
        w = data.w.view(32,200).cpu().numpy()
        w = w[:,:100] + 1j*w[:,100:]
        #================================dis=======================
        dis = data.dis.cpu().numpy()
        out_dis = np.zeros_like(dis)
        while mask_list.sum() > n:
            mask = torch.multinomial(mask_list, n, replacement=False)
            mask_list[mask] = 0
            data1 = Data()
            data1.pos = data.pos[mask]
            data1.normal = data.normal[mask]
            data1.batch = torch.zeros(len(data1.pos),dtype=torch.int64)
            data1 = data1.to(device)
            out = model_dis(data1)
            out_dis[mask] = out.cpu().numpy() 

        mask = torch.multinomial(mask_list, n, replacement=True)
        data1 = Data()
        data1.pos = data.pos[mask]
        data1.normal = data.normal[mask]
        data1.batch = torch.zeros(len(data1.pos),dtype=torch.int64)
        data1 = data1.to(device)
        out = model_dis(data1)
        out_dis[mask] = out.cpu().numpy() 


        modes_dict = {i:[] for i in range(freq.resolution)}
        for i,omega in enumerate(omegas):
            modes_dict[freq.omega2index(omega)].append(i)
        for i in range(23,24):
            if data.s[0,i] == 0:
                continue
            coeff = poles_coeffs[modes_dict[i]]
            export(f'{output_dir}/gt{i}.msh', grid_function=GridFunction(mesh.dp0_space, coefficients=dis[:,i]))
            export(f'{output_dir}/predict{i}.msh', grid_function=GridFunction(mesh.dp0_space, coefficients=out_dis[:,i]))
            export(f'{output_dir}/{i}_sphere.msh',grid_function=pole_matrix.get_grid_function(sphere, out_w[i], freq.index2omega(i)/SPEED_OF_SOUND))
            export(f'{output_dir}/{i}_sphere_gt.msh',grid_function=pole_matrix.get_grid_function(sphere, w[i], freq.index2omega(i)/SPEED_OF_SOUND))
            export(f'{output_dir}/{i}_sphere_gt2.msh',grid_function=pole_matrix.get_grid_fun_from_list(sphere, coeff, freq.index2omega(i)/SPEED_OF_SOUND))

    work('/home/jxt/ssd_dataset/test/bowl_0080.pt')
test()



# from analysis.voxelize import Hexa_model
# from analysis.BEM import boundary_mesh
# from bempp.api import export, GridFunction

# from mpl_toolkits.mplot3d import Axes3D 
# import matplotlib.pyplot as plt
# import numpy as np



# filename = 'bottle_0338.off'
# vox = Hexa_model(filename)
# fig = plt.figure()
# ax = fig.gca(projection='3d')
# ax.voxels(vox.voxel_grid.reshape(32,32,32))

# plt.show()
# # vox.create_tetra_and_boundary()
# # mesh = boundary_mesh(vox.vertices, vox.boundary_faces)
# # export('test.msh',grid=mesh.grid)
