
import bempp.api
from bempp.api import export, GridFunction
from analysis.FEM import FEM_model, Material
from analysis.BEM import boundary_mesh, PolesMatrix, SPEED_OF_SOUND, AIR_DENSITY
import numpy as np
import os
from scipy.linalg import lstsq
from tqdm import tqdm
from glob import glob


def work(file_list):
    sphere = boundary_mesh(grid=bempp.api.shapes.sphere(h=0.02,r = 0.2))
    pole_matrix = PolesMatrix()
    pole_matrix.sample_points(0.1)
    pole_matrix.assemble_matrix()
    print('pole matrix initialized')

    for dirname in tqdm(file_list):
        if os.path.exists(dirname+'/displacements.npy'):
            continue
        vertices = np.load(dirname+'/vertices.npy')
        print(dirname)
        boundary_faces = np.load(dirname+'/boundary_faces.npy')
        print(boundary_faces.shape)
        tets = np.load(dirname + '/tets.npy')
        current_model = boundary_mesh(vertices=vertices, faces=boundary_faces)
        #=====================FEM modal analysis=================
        fem = FEM_model(vertices, tets)
        fem.set_material(Material.Iron)
        fem.create_matrix()
        fem.compute_modes(min_freq=20,max_freq=20000)
        print(len(fem.omega_d))
        if len(fem.omega_d) < 5:
            continue
        if fem.omega_d[0]/(2*np.pi) < 500:
            continue
        #=====================save data==========================
        export(dirname+'/mesh.msh', grid=current_model.grid)
        np.save(dirname+'/face_centers', current_model.face_centers())
        np.save(dirname+'/face_normals', current_model.normals())
        np.save(dirname+'/omegas', fem.omega_d)
        modes_num = len(fem.vals)
        poles_coeffs = np.zeros((modes_num, pole_matrix.poles.pole_number), dtype = np.complex)
        displacements = np.zeros((modes_num, len(boundary_faces)))
        for i in range(modes_num):
            omega = fem.omega_d[i]
            k_ = omega / SPEED_OF_SOUND
            freq_idx = pole_matrix.wavenumber2index(k_)
            k = pole_matrix.wave_numbers[freq_idx]
            #=================BEM=======================
            displacement = fem.vecs[:,i].reshape(-1,3)
            displacement = displacement[boundary_faces].mean(1)
            displacement = (displacement*current_model.normals()).sum(1)
            displacements[i] = displacement
            neumann_coeff = AIR_DENSITY*omega**2*displacement
            neumann_fun =  GridFunction(current_model.dp0_space, coefficients=np.asarray(neumann_coeff))
            current_model.set_wave_number(k)
            current_model.set_neumann_fun(neumann_fun)
            current_model.ext_neumann2dirichlet()
            #==================least square method=================
            b = current_model.points_dirichlet(pole_matrix.points)
            A = pole_matrix.all_matrix[freq_idx]
            weights, res, _, _ = lstsq(A,b)
            poles_coeffs[i] = weights
        np.save(dirname+'/poles_coeffs', poles_coeffs)
        np.save(dirname+'/displacements', displacements)

import sys
if __name__ == "__main__":
    file_list = glob('dataset/' + sys.argv[1] + '/*/*')
    print('file list gotten')
    work(file_list)