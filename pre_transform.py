from analysis import voxel, FEM_cuda
from visualization.viewer import Viewer
from visualization.audio_player import Player
from glob import glob
import numpy as np 
from time import time
from tqdm import tqdm
import os
import scipy
from scipy.sparse.linalg import eigsh
import open3d as o3d
def hz2mel(f):
    m = 2595*np.log10(1+f/700)
    return m

def mel2hz(m):
    f = (10**(m/2595)-1)*700
    return f

resolution = 32
m_min = hz2mel(100)
m_max = hz2mel(10000)
spacing = (m_max - m_min)/resolution

def hz2index(f):
    m = hz2mel(f)
    return int((m-m_min)/spacing)

def index2hz(i):
    m = m_min + (i+0.5)*spacing
    return mel2hz(m)

def model_synthesis(file_list, root):
    for filename in tqdm(file_list):
        dirname = root + os.path.basename(os.path.dirname(filename))
        os.makedirs(dirname, exist_ok=True)

        if os.path.exists(dirname + '/vecs_norm.npy'):
            continue
        #====================voxelize=======================
        vox = voxel.VOX(filename, 32)
        vox.create_tetra_mesh_cuda(0.2)
        vox.create_triangle_mesh()
        o3d.io.write_triangle_mesh(dirname + '/mesh.ply', vox.triangle_mesh)
        
        #===============FEM matrix extract==================
        fem = FEM_cuda.FEM_model(vox.tetra_mesh.vertices, vox.tetra_mesh.tetras)
        fem.set_material(0)
        fem.compute_matrix()
        np.save(dirname+'/vertices', fem.vertices)
        scipy.sparse.save_npz(dirname + '/stiff_matrix', fem.stiff_matrix)
        scipy.sparse.save_npz(dirname + '/mass_matrix', fem.mass_matrix)

        #===============model analysis======================
        min_freq = 100
        max_freq = 10000 
        modes_num = 50
        sigma = ((2*np.pi*max_freq)**2 + (2*np.pi*min_freq)**2)/2
        vals, vecs = eigsh(fem.stiff_matrix, k=modes_num, M=fem.mass_matrix,which='LM',sigma=sigma)
        while max(vals) < (2*np.pi*max_freq)**2 :
            modes_num += 50
            vals, vecs = eigsh(fem.stiff_matrix, k=modes_num, M=fem.mass_matrix,which='LM',sigma=sigma)

        #================normalize==========================
        alpha=2E-6
        beta=60.0
        c = (alpha*vals + beta)
        omega = np.sqrt(vals)
        valid = (1 - c**2/(omega**2*4) > 0)
        vals = vals[valid]
        vecs = vecs[:,valid]
        c = (alpha*vals + beta)
        omega = np.sqrt(vals)
        omega_d = omega*np.sqrt(1 - c**2/(omega**2*4))

        vecs_normalized = np.zeros((vecs.shape[0],resolution))
        for i in range(len(vals)):
            idx = hz2index(omega_d[i]/(2*np.pi))
            if idx < 32 and idx >= 0:
                vecs_normalized[:,idx] += vecs[:,i]
        vecs_normalized = np.abs(vecs_normalized)

        np.save(dirname + '/vecs_norm', vecs_normalized)
        np.save(dirname + '/vals', vals)
        np.save(dirname + '/vecs', vecs)



if __name__ == "__main__":
    dir_list = glob('D:/abcDataset/abc/*/*.obj')[:50000]
    test_list = dir_list[:len(dir_list)//5]
    train_list = dir_list[len(dir_list)//5:]
    valid_list = train_list[:len(train_list)//5]
    train_list = train_list[len(train_list)//5:]

    model_synthesis(train_list, 'G:/dataset/train/')
    model_synthesis(test_list, 'G:/dataset/test/')
    model_synthesis(valid_list, 'G:/dataset/valid/')

