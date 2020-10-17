from analysis import voxel,FEM, FEM_cuda
from visualization.viewer import Viewer
from visualization.audio_player import Player
from glob import glob
import numpy as np 
from time import time
import sys
import argparse
if __name__ == "__main__":
    # ==========================args parser===============================
    parser = argparse.ArgumentParser(description='Modal synthesis')
    parser.add_argument('--idx', type=int, default=0, help='Mesh directory index')
    parser.add_argument('--mat', type=int, default=4, help='Material index')
    parser.add_argument('--scale', type=float, default=0.2, help='Scale (meter)')
    parser.add_argument('--mesh', type=str, default='D:/abcDataset/abc', help='Mesh dataset directory (abc dataset)')
    args = parser.parse_args()

    if not args.idx:
        parser.error('Index required, add --idx ')
    # ============================pre-process=============================
    filename = glob(f'{args.mesh}/{args.idx:08d}/*')[0]
    print(filename)
    vox = voxel.VOX(filename, 32)
    vox.create_tetra_mesh_cuda(args.scale)
    vox.create_triangle_mesh()
    print('tetra mesh generated')
    fem = FEM_cuda.FEM_model(vox.tetra_mesh.vertices, vox.tetra_mesh.tetras)
    fem.set_material(args.mat)
    fem.compute_matrix()
    print('matrix computed')
    fem.compute_modes(max_freq=20000)
    print('modes computed')

    
    # ==========================run-time demo============================
    mesh = vox.triangle_mesh
    audio_player = Player()
    vis = Viewer()
    vis.load_mesh(mesh.vertices, mesh.triangles, mesh.vertex_normals)

    def click_callback(index):
        amp = np.zeros(len(fem.vals))
        if index >= len(vis.faces):
            print('out of index')
            return
        for vid in vis.faces[index]:
            amp += vis.normals[vid].dot(fem.vecs[3*vid:3*(vid+1),:])
        amp = np.abs(amp)
        valid = (1 - (fem.alpha*fem.vals + fem.beta)**2/(fem.vals*4) > 0)
        amp = amp[valid]
        vals = fem.vals[valid]
        c = (fem.alpha*vals + fem.beta)
        omega = np.sqrt(vals)
        omega_d = omega*np.sqrt(1 - c**2/(omega**2*4))
        amp = amp / omega_d
        print(omega_d/(2*np.pi))
        audio_player.play(amp*1e3,omega_d/(2*np.pi),c)

    vis.connect_click(click_callback)
    vis.run()

    
    