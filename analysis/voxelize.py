import open3d as o3d
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
from analysis.cpp import boundary

mod = SourceModule(open('analysis/cuda/voxelize.cu').read())
cuda_voxelizer = mod.get_function("voxelize")
cuda_tetraer = mod.get_function('tetra')

def center_scale(scale):
    def transform(vertices):
        box_lens = vertices.max(0) - vertices.min(0)
        center = (vertices.max(0) + vertices.min(0))/2
        return (vertices - center) / max(box_lens) * scale
    return transform

class vox_model():
    def __init__(self, vertices, tets, faces):
        self.vertices = vertices
        self.tets = tets
        self.boundary_faces = faces
        self.boundary_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), 
                                            o3d.utility.Vector3iVector(faces))
    def set_transform(self,lst):
        for transform in lst:
            self.vertices = transform(self.vertices)

    def save_boundary_mesh(self, filename):
        o3d.io.write_triangle_mesh(filename, self.boundary_mesh)

class Hexa_model(vox_model):
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

    def create_tetra_and_boundary(self):
        self.create_tetra_mesh()
        self.create_boundary()

    def create_tetra_mesh(self):
        flags = self.voxel_grid.copy()
        flags = np.asarray(boundary.flood_dfs(flags, self.res)).reshape(self.res+2,self.res+2,self.res+2)
        flags = flags[1:-1,1:-1,1:-1].reshape(-1)
        flags = 1 - flags
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
        self.tetra_mesh = tetra_mesh


    def create_boundary(self):
        if self.vertices is None:
            print('Tetra mesh is needed!')
            return
        faces = np.asarray(boundary.create(self.voxel_grid, self.vertices, self.tets, self.res)).reshape(-1,3)
        mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(self.vertices), 
                                            o3d.utility.Vector3iVector(faces))
        self.boundary_mesh = mesh
        self.boundary_faces = faces


# class Plate_model(vox_model):
#     def __init__(self, res):
#         self.res = res

#     def create_tetra_and_boundary(self, delta = 0):
#         res = self.res
#         vertices  = np.zeros((0,3),dtype=np.float64)
#         tets = np.zeros((0,4),dtype=np.int32)
#         faces = np.zeros((0,3), dtype=np.int32)
#         for i in range(res):
#             theta1 = np.pi * 2 / res * ((i+delta) % res)
#             theta2 = np.pi * 2 / res * ((i+1 + delta) % res)
#             thickness = 0.1
#             vs = np.array([
#                 [0,0,0],[0,0,thickness],[np.cos(theta1),np.sin(theta1),0],[np.cos(theta2),np.sin(theta2),0],
#                 [np.cos(theta1),np.sin(theta1),thickness],[np.cos(theta2),np.sin(theta2),thickness],
#                 [np.cos(theta1)/2,np.sin(theta1)/2,0],[np.cos(theta2)/2,np.sin(theta2)/2,0],
#                 [np.cos(theta1)/2,np.sin(theta1)/2,thickness],[np.cos(theta2)/2,np.sin(theta2)/2,thickness],
#             ]).astype(np.float64)
#             ts = np.array([
#                 [0,1,6,7],[1,6,7,9],[1,6,8,9],
#                 [5,6,7,9],[6,4,8,5],[6,9,8,5],
#                 [4,5,2,6],[5,2,3,6],[5,7,3,6]
#             ]).astype(np.int32)
#             fs = np.array([
#                 [8,9,1],[8,5,9],[4,5,8],
#                 [7,6,0],[3,2,6],[7,3,6],
#                 [2,5,4],[3,5,2]
#             ])
#             idx = len(vertices)
#             vertices  = np.concatenate([vertices, vs])
#             tets = np.concatenate([tets, ts + idx])
#             faces = np.concatenate([faces, fs + idx])

#         tetra_mesh = o3d.geometry.TetraMesh(o3d.utility.Vector3dVector(vertices), 
#                                                 o3d.utility.Vector4iVector(tets))
#         triangle_mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(vertices), 
#                                                 o3d.utility.Vector3iVector(faces))
#         tetra_mesh.remove_duplicated_vertices()
#         triangle_mesh.remove_duplicated_vertices()
#         triangle_mesh.compute_vertex_normals()
#         triangle_mesh.scale(1 / np.max(triangle_mesh.get_max_bound() - triangle_mesh.get_min_bound()), center=triangle_mesh.get_center())
#         triangle_mesh.translate(-triangle_mesh.get_center())
#         self.boundary_mesh = triangle_mesh
#         self.vertices = np.asarray(tetra_mesh.vertices)
#         self.tets = np.asarray(tetra_mesh.tetras)
#         self.boundary_faces = np.asarray(triangle_mesh.triangles)
        
