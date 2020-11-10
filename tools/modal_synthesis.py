from analysis.voxelize import Hexa_model, center_scale
from analysis.FEM import FEM_model
from visualization.viewer import Viewer
from visualization.audio_player import Player
from glob import glob
import numpy as np 
from time import time
import sys
import open3d as o3d

if __name__ == "__main__":
    filename = sys.argv[1]
    print(f'Modal synthesis of {filename}:')
    vox = Hexa_model(filename)
    vox.create_tetra_and_boundary()
    vox.set_transform([center_scale(0.2)])
    print(vox.vertices.max())
    print('tetra mesh and boundary triangle mesh generated')

    
    fem = FEM_model(vox.vertices, vox.tets)
    fem.set_material(0)
    fem.create_matrix()
    fem.compute_modes(min_freq=100,max_freq=10000)
    print('modal data generated')
    
    # ==========================run-time demo============================
    mesh = vox.boundary_mesh
    mesh.compute_vertex_normals()

    vis = Viewer()
    vis.load_mesh(vox.vertices, mesh.triangles, mesh.vertex_normals)

    audio_player = Player()
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
        audio_player.play(amp*1e3,omega_d/(2*np.pi),c)

    vis.connect_click(click_callback)
    vis.run()

    
    
