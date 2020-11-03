import bempp.api
from bempp.api import export, GridFunction
from analysis.FEM import FEM_model
from analysis.BEM import boundary_mesh, PolesMatrix, SPEED_OF_SOUND, AIR_DENSITY
from analysis import frequency
import numpy as np
import os
from scipy.linalg import lstsq
from tqdm import tqdm
from glob import glob
import scipy

freq = frequency.FrequencyScale(32)

dirname = 'dataset/0/00000008'
vertices = np.load(dirname+'/vertices.npy')
boundary_faces = np.load(dirname+'/boundary_faces.npy')
current_model = boundary_mesh(vertices=vertices, faces=boundary_faces)
displacements = np.load(dirname+'/displacements.npy')
poles_coeffs = np.load(dirname+'/poles_coeffs.npy')
omegas = np.load(dirname+'/omegas.npy')
modes_dict = {i:[] for i in range(freq.resolution)}
print(poles_coeffs.dtype)
main_displacements = np.zeros(freq.resolution,dtype=np.complex64)
main_coeffs = np.zeros((freq.resolution,poles_coeffs.shape[1]), dtype=np.complex64)
for i,omega in enumerate(omegas):
    modes_dict[freq.omega2index(omega)].append(i)
for i in range(freq.resolution):
    idxs = modes_dict[i]
    if len(idxs) == 0:
        continue
    displace = displacements[idxs]
    print(displace.shape)
    coeff = poles_coeffs[idxs]
    print(coeff.shape)
    A = (displace.T[...,np.newaxis]*coeff).sum(-2)
    c = A.sum(0)
    A = A - c
    U,S,V = scipy.sparse.linalg.svds(A,k=1)
    # print(V,U)
    # print(np.abs(V*V).sum())
    # print(np.abs(U*U).sum())
    

    
    
                           
# import torch
# a = torch.Tensor([[1,1,1,1,2,2,2,2]])
# A = torch.cat((a,a*2,a*3,a*4,a*5,a*6,a*7))
# c = A.mean(0)
# A = A - c
# U,S,V = torch.pca_lowrank(A, 2)
# print(U)
# print(S)
# print(V)
