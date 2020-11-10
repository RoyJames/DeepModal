from analysis import frequency
import numpy as np
import os
from tqdm import tqdm
from glob import glob
import scipy
from torch_geometric.data import Data
import torch
import bempp.api
from bempp.api import GridFunction, export
from analysis.BEM import boundary_mesh,PolesMatrix,SPEED_OF_SOUND


freq = frequency.FrequencyScale(32)
pole_matrix = PolesMatrix()
pole_matrix.sample_points(0.1)
pole_matrix.assemble_matrix()
print('pole matrix initialized')


def preprocess(dirname):
    displacements = np.load(dirname+'/displacements.npy')
    poles_coeffs = np.load(dirname+'/poles_coeffs.npy')
    omegas = np.load(dirname+'/omegas.npy')
    
    mesh = boundary_mesh(np.load(dirname+'/vertices.npy'),np.load(dirname+'/boundary_faces.npy'))

    modes_dict = {i:[] for i in range(freq.resolution)}
    for i,omega in enumerate(omegas):
        modes_dict[freq.omega2index(omega)].append(i)

    for i in range(23,24):
        idxs = modes_dict[i]
        if len(idxs) == 0:
            continue
        omega = freq.index2omega(i)
        k = omega / SPEED_OF_SOUND
        displace = displacements[idxs]
        displace = abs(displace)
        coeff = poles_coeffs[idxs]
        print(coeff)
        # A = (displace.T[...,np.newaxis]*coeff).sum(-2)
        # c = A.sum(0)
        # print(c.shape)
        # A = A - c
        # U,S,V = scipy.sparse.linalg.svds(A,k=2)
        # print(U.shape)
        # export(f'modedata/{i}_1.msh',grid_function=GridFunction(mesh.dp0_space, coefficients=U[:,0]))
        # export(f'modedata/{i}_2.msh',grid_function=GridFunction(mesh.dp0_space, coefficients=U[:,1]))

        for j,displace_ in enumerate(displace):
            export(f'modedata/{i}_dis{j}.msh',grid_function=GridFunction(mesh.dp0_space, coefficients=displace_))
        export(f'modedata/{i}_dis_all.msh',grid_function=GridFunction(mesh.dp0_space, coefficients=displace.sum(0)))
        export(f'modedata/{i}_dis_all_square.msh',grid_function=GridFunction(mesh.dp0_space, coefficients=(displace**2).sum(0)**0.5))

        sphere = boundary_mesh(grid=bempp.api.shapes.sphere(h=0.02,r = 0.2))
        for j,coeff_ in enumerate(coeff):
            export(f'modedata/{i}_coeff{j}.msh',grid_function=pole_matrix.get_grid_function(sphere, coeff_, k))
        export(f'modedata/{i}_coeff_all.msh',grid_function=pole_matrix.get_grid_fun_from_list(sphere, coeff, k))


import sys
preprocess(sys.argv[1])