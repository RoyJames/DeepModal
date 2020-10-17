import open3d as o3d
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from analysis.cpp import boundary

mod = SourceModule(open('analysis/cuda/voxelize.cu').read())
cuda_voxelizer = mod.get_function("voxelize")
cuda_tetraer = mod.get_function('tetra')
class VOX:
    def __init__(self, filename = None, res = 32):
        dest = np.zeros(res*res*res,dtype= np.int32)
        mesh = None
        if filename is not None:
            mesh = self.normalize_mesh(o3d.io.read_triangle_mesh(filename))
            vertices = np.asarray(mesh.vertices).astype(np.float32)
            triangles = np.asarray(mesh.triangles).astype(np.int32)
            cuda_voxelizer(
                drv.Out(dest), np.int32(res), drv.In(vertices),
                drv.In(triangles), np.int32(len(triangles)),
                block=(res,res,1), grid=(res,1)
                )
        self.mesh = mesh
        self.voxel_grid = dest
        self.res = res
        self.vertices = None
        self.tets = None 
        self.boundary_mesh = None
        self.boundary_faces = None
    
    def normalize_mesh(self, mesh):
        mesh.compute_vertex_normals()
        mesh.scale(1 / np.max(mesh.get_max_bound() - mesh.get_min_bound()), center=mesh.get_center())
        mesh.translate(-mesh.get_center())
        return mesh

    def create_tetra_mesh(self):
        flags = self.voxel_grid.copy()
        tets_num = flags.sum()
        flags[flags == 1] = np.arange(tets_num ,dtype=np.int32) + 1
        vertices  = np.zeros(tets_num*8*3,dtype=np.float32)
        tets = np.zeros(tets_num*5*4, dtype=np.int32)
        cuda_tetraer(
            drv.Out(tets), drv.Out(vertices), np.int32(self.res), drv.In(flags),
            block=(self.res,self.res,1), grid=(self.res,1)
        )
        tetra_mesh = o3d.geometry.TetraMesh(o3d.utility.Vector3dVector(vertices.reshape(-1,3).astype(np.float64)), 
                                            o3d.utility.Vector4iVector(tets.reshape(-1,4)))
        tetra_mesh.remove_duplicated_vertices()
        self.vertices = np.asarray(tetra_mesh.vertices)
        self.tets = np.asarray(tetra_mesh.tetras)


    def create_boundary(self):
        if self.vertices is None:
            print('Tetra mesh is needed!')
            return
        faces = np.asarray(boundary.create(self.voxel_grid, self.vertices, self.tets, self.res)).reshape(-1,3)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self.vertices), 
                                            o3d.utility.Vector3iVector(faces))
        self.boundary_mesh = mesh
        self.boundary_faces = faces


