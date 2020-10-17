from analysis import voxel, FEM_cuda
from visualization import check
from glob import glob
import numpy as np 
from time import time
from tqdm import tqdm
import os
import scipy
from scipy.sparse.linalg import eigsh
import open3d as o3d

def plate_test():
    dirname = 'testdata'
    os.makedirs(dirname, exist_ok=True)

    #==================plane mesh generated===================
    vertices  = np.zeros((0,3),dtype=np.float64)
    tets = np.zeros((0,4),dtype=np.int32)
    faces = np.zeros((0,3), dtype=np.int32)
    res = 20
    delta = 4
    for i in range(res):
        theta1 = np.pi * 2 / res * ((i+delta) % res)
        theta2 = np.pi * 2 / res * ((i+1 + delta) % res)
        thickness = 0.1
        vs = np.array([
            [0,0,0],[0,0,thickness],[np.cos(theta1),np.sin(theta1),0],[np.cos(theta2),np.sin(theta2),0],
            [np.cos(theta1),np.sin(theta1),thickness],[np.cos(theta2),np.sin(theta2),thickness],
            [np.cos(theta1)/2,np.sin(theta1)/2,0],[np.cos(theta2)/2,np.sin(theta2)/2,0],
            [np.cos(theta1)/2,np.sin(theta1)/2,thickness],[np.cos(theta2)/2,np.sin(theta2)/2,thickness],
        ]).astype(np.float64)
        ts = np.array([
            [0,1,6,7],[1,6,7,9],[1,6,8,9],
            [5,6,7,9],[6,4,8,5],[6,9,8,5],
            [4,5,2,6],[5,2,3,6],[5,7,3,6]
        ]).astype(np.int32)
        fs = np.array([
            [8,9,1],[8,5,9],[4,5,8],
            [7,6,0],[3,2,6],[7,3,6],
            [2,5,4],[3,5,2]
        ])
        
        idx = len(vertices)
        vertices  = np.concatenate([vertices, vs])
        tets = np.concatenate([tets, ts + idx])
        faces = np.concatenate([faces, fs + idx])

    tetra_mesh = o3d.geometry.TetraMesh(o3d.utility.Vector3dVector(vertices), 
                                            o3d.utility.Vector4iVector(tets))
    triangle_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), 
                                            o3d.utility.Vector3iVector(faces))
    tetra_mesh.remove_duplicated_vertices()
    triangle_mesh.remove_duplicated_vertices()
    triangle_mesh.compute_vertex_normals()
    triangle_mesh.scale(1 / np.max(triangle_mesh.get_max_bound() - triangle_mesh.get_min_bound()), center=triangle_mesh.get_center())
    triangle_mesh.translate(-triangle_mesh.get_center())
    print(len(triangle_mesh.vertices))
    print(len(tetra_mesh.vertices))
    o3d.io.write_triangle_mesh(dirname + '/mesh.ply', triangle_mesh)

    #===============FEM matrix extract==================
    fem = FEM_cuda.FEM_model(tetra_mesh.vertices, tetra_mesh.tetras)
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
    np.save(dirname + '/vals', vals)
    np.save(dirname + '/vecs', vecs)
    #====================check==============================
    checker = check.check('./testdata/','/vecs.npy', '/vecs.npy')
    checker.dir = './testdata/'
    checker.load_mesh()
    checker.load_modes()
    checker.v2.window.hide()
    checker.main_window.hide()
    checker.run()

if __name__ == "__main__":
    #plate_test()
    M = scipy.sparse.load_npz('testdata/mass_matrix.npz').todense()
    K = scipy.sparse.load_npz('testdata/stiff_matrix.npz').todense()
    vals = np.load('testdata/vals.npy')
    vecs1 = np.load('testdata/vecs0.npy')
    vecs2 = np.load('testdata/vecs.npy')
    u1 = vecs1[:,0]
    u2 = vecs2[:,0]
    lda = vals[0]
    print(u2.dot(K))
    print(u2.dot(K) - lda*u2.dot(M))
