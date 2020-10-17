from analysis import voxel,FEM, FEM_cuda
from visualization.viewer import Viewer
from visualization.audio_player import Player
from glob import glob
import numpy as np 
from time import time
import argparse

def hz2mel(f):
    m = 2595*np.log10(1+f/700)
    return m

resolution = 32
m_min = hz2mel(100)
m_max = hz2mel(10000)
spacing = (m_max - m_min)/resolution

def mel2hz(m):
    f = (10**(m/2595)-1)*700
    return f

def index2hz(i):
    m = m_min + (i+0.5)*spacing
    return mel2hz(m)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Modal synthesis based on approximate data')
    parser.add_argument('--idx', type=int, help='Data directory index')
    parser.add_argument('--mat', type=int, default=4, help='Material index')
    parser.add_argument('--scale', type=float, default=1, help='Scale rate')
    parser.add_argument('--mesh', type=str, default='D:/abcDataset/abc', help='Mesh dataset directory (abc dataset)')
    parser.add_argument('--data', type=str, default='G:/dataset', help='Data directory (modal data)')
    args = parser.parse_args()
    mesh_file = glob(f'{args.mesh}/{args.idx:08d}/*')[0]
    data_dir = f'{args.data}/{args.idx:08d}'

    print(mesh_file)
    #=======================load data===========================
    vox = voxel.VOX(mesh_file, 32)
    vox.create_tetra_mesh_cuda(0.2)
    vox.create_triangle_mesh()
    mesh = vox.triangle_mesh
    vecs_normalized = np.load(data_dir + '/vecs_norm.npy')
    mat_default = FEM_cuda.FEM_model()
    mat_default.set_material(4)
    mat = FEM_cuda.FEM_model()
    mat.set_material(args.mat)
    k1 = mat.youngs/mat_default.youngs*mat_default.density/mat.density
    k2 = mat_default.density/mat.density

    #=======================post-process=========================

    f_ = index2hz(np.arange(resolution))
    c_ = 2.*(1 - np.sqrt(
                1 - mat_default.alpha*
                (mat_default.beta + mat_default.alpha * (f_*2*np.pi)**2)
                ) 
            )/mat_default.alpha
    wd_ = f_*2*np.pi
    w_ = np.sqrt(wd_*wd_ + c_*c_/4)
    val_ = w_*w_
    val = val_ * k1
    w = np.sqrt(val) /args.scale
    c = mat.alpha*val + mat.beta
    wd = np.sqrt(w*w - c*c/4)
    f = wd/(2*np.pi)
    a_k =  np.sqrt(k2/args.scale**3) / wd

    #======================run-time demo=========================
    audio_player = Player()
    vis = Viewer()
    vis.load_mesh(mesh.vertices, mesh.triangles, mesh.vertex_normals)
    def click_callback(index):
        amp = np.zeros(vecs_normalized.shape[1])
        if index >= len(vis.faces):
            print('out of index')
            return
        for vid in vis.faces[index]:
            amp += vis.normals[vid].dot(vecs_normalized[3*vid:3*(vid+1),:])
        amp = np.abs(amp)*a_k 
        audio_player.play(amp*1e3, f, c)

    vis.connect_click(click_callback)
    vis.run()
    

    