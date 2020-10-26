
import bempp.api
from bempp.api import export, GridFunction
from bempp.api.operators import potential, boundary
from analysis.voxelize import Hexa_model, Plate_model, center_scale
from analysis.FEM import FEM_model
import numpy as np
import open3d as o3d
import os

#bempp.api.LOGGER.propagate = False

class boundary_mesh():
    def __init__(self, vertices = None, faces = None, grid = None):
        if grid is not None:
            vertices = grid.vertices.T
            faces = grid.elements.T

        self.vertices = vertices
        self.faces = faces
        self.grid = bempp.api.Grid(vertices.T.astype(np.float64), 
                        faces.T.astype(np.uint32))
        self.dp0_space = bempp.api.function_space(self.grid, "DP", 0)
        self.p1_space = bempp.api.function_space(self.grid, "P", 1)
        self.dirichlet_fun = self.neumann_fun = None
    
    def face_centers(self):
        return self.vertices[self.faces].mean(1)

    def set_wave_number(self,k):
        self.k = k
    
    def set_neumann_fun(self,neumann_fun):
        self.neumann_fun = neumann_fun

    def set_dirichlet_fun(self, dirichlet_fun):
        self.dirichlet_fun = dirichlet_fun

    def ext_neumann2dirichlet(self):
        self.double = boundary.helmholtz.double_layer(
                        self.p1_space, self.p1_space, self.p1_space, self.k, device_interface='opencl')
        self.single = boundary.helmholtz.single_layer(
                        self.dp0_space, self.p1_space, self.p1_space, self.k, device_interface='opencl')
        self.identity = boundary.sparse.identity(
                        self.p1_space, self.p1_space, self.p1_space)
        left_side = (-.5*self.identity+self.double)
        right_side = self.single*self.neumann_fun
        dirichlet_fun, info, _ = bempp.api.linalg.gmres(left_side, right_side,tol=1e-6, return_residuals=True)
        self.dirichlet_fun = dirichlet_fun

    def points_dirichlet(self, points):
        potential_single = potential.helmholtz.single_layer(self.dp0_space, points.T, self.k)
        potential_double = potential.helmholtz.double_layer(self.p1_space, points.T, self.k)
        dirichlet = -potential_single*self.neumann_fun + potential_double*self.dirichlet_fun 
        return dirichlet.reshape(-1)



def test():
    scale = 0.2
    vox = Plate_model(200)
    vox.create_tetra_and_boundary()
    vox.set_transform([center_scale(scale)])
    fem = FEM_model(vox.vertices, vox.tets)
    fem.set_material(0)
    fem.create_matrix()
    fem.compute_modes()

    print('==========modal data=============')
    print(fem.vals.shape)
    print(fem.vecs.shape)
    print(fem.vertices.max(), fem.vertices.min())

    sphere = boundary_mesh(grid=bempp.api.shapes.sphere(h=0.01,r = 0.1*3**(0.5)))
    current_model = boundary_mesh(vertices=vox.vertices, faces=vox.boundary_faces)

    for i in range(len(fem.vals)):
        c = 343
        omega = fem.vals[i]
        displacement = fem.vecs[:,i]
        k = omega / c
        displacement = displacement.reshape(-1,3)
        displacement = displacement[vox.boundary_faces].mean(1)
        c = (displacement*current_model.grid.normals).sum(1)
        neumann_fun =  GridFunction(current_model.dp0_space, coefficients=c)

        current_model.set_wave_number(k)
        current_model.set_neumann_fun(neumann_fun)
        current_model.ext_neumann2dirichlet()
        
        export(f'./modedata/{i}stMode.msh',grid_function=current_model.neumann_fun)
        export(f'./modedata/{i}stMode_d.msh',grid_function=current_model.dirichlet_fun)
        
        coeffs = current_model.points_dirichlet(sphere.face_centers())
        print(coeffs.shape)
        print(sphere.faces.shape)
        export(f'./modedata/{i}st_sphere.msh', grid_function=GridFunction(sphere.dp0_space, 
                                                                coefficients=coeffs))

def check():
    r_larger = 0.1
    r_smaller = 0.1*3**(0.5)
    sphere = boundary_mesh(grid=bempp.api.shapes.sphere(h=0.01,r = 0.1*3**(0.5)))
    current_model = boundary_mesh(grid=bempp.api.shapes.sphere(h=0.01,r = 0.1))
    c = 343
    omega = 1000*6.28
    k = omega / c
    displacement = displacement.reshape(-1,3)
    displacement = displacement[vox.boundary_faces].mean(1)
    c = (displacement*current_model.grid.normals).sum(1)
    neumann_fun =  GridFunction(current_model.dp0_space, coefficients=c)

    current_model.set_wave_number(k)
    current_model.set_neumann_fun(neumann_fun)
    current_model.ext_neumann2dirichlet()
    
    export(f'./modedata/{i}stMode.msh',grid_function=current_model.neumann_fun)
    export(f'./modedata/{i}stMode_d.msh',grid_function=current_model.dirichlet_fun)
    
    coeffs = current_model.points_dirichlet(sphere.face_centers())
    print(coeffs.shape)
    print(sphere.faces.shape)
    export(f'./modedata/{i}st_sphere.msh', grid_function=GridFunction(sphere.dp0_space, 
                                                            coefficients=coeffs))