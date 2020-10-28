
import bempp.api
from bempp.api import export, GridFunction
from bempp.api.operators import potential, boundary
from analysis.voxelize import Hexa_model, Plate_model, center_scale
from analysis.FEM import FEM_model
from analysis import special
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
        self.adjoint_double = boundary.helmholtz.adjoint_double_layer(
                        self.dp0_space, self.dp0_space, self.p1_space, self.k, device_interface='opencl')
        self.hyper_single = boundary.helmholtz.hypersingular(
                        self.p1_space, self.dp0_space, self.p1_space, self.k, device_interface='opencl')
        self.identity = boundary.sparse.identity(
                        self.dp0_space, self.dp0_space, self.p1_space)
        left_side = self.hyper_single
        right_side = (-0.5*self.identity-self.adjoint_double)*self.neumann_fun
        dirichlet_fun, info, _ = bempp.api.linalg.gmres(left_side, right_side,tol=1e-6, return_residuals=True)
        self.dirichlet_fun = dirichlet_fun
        export('./modedata/left.msh', grid_function = left_side*dirichlet_fun)
        export('./modedata/right.msh', grid_function = right_side)

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
    r_in = 0.1
    r_out = 0.1*3**(0.5)
    sphere = boundary_mesh(grid=bempp.api.shapes.sphere(h=0.01,r = r_out))
    current_model = boundary_mesh(grid=bempp.api.shapes.sphere(h=0.01,r = r_in))
    c = 343
    omega = 1000*6.28
    k = omega / c

    scale = 10
    poles = special.Multipole(scale)
    weights = np.zeros(poles.pole_number)
    weights[0] = 0.3
    weights[10] = 0.3
    weights[5] = 0.3
    neumann_coeff = []
    dirichlet_coeff = []
    for point in current_model.face_centers():
        poles.reset(k,point)
        poles.dirichlet_reset()
        poles.neumann_reset()
        neumann_coeff.append((poles.neumann*weights).sum())
        dirichlet_coeff.append((poles.dirichlet*weights).sum())

    neumann_fun =  GridFunction(current_model.dp0_space, coefficients=np.asarray(neumann_coeff))
    dirichlet_fun = GridFunction(current_model.dp0_space, coefficients=np.asarray(dirichlet_coeff))


    current_model.set_wave_number(k)
    current_model.set_neumann_fun(neumann_fun)
    current_model.ext_neumann2dirichlet()
    
    export(f'./modedata/test_N.msh',grid_function=current_model.neumann_fun)
    export(f'./modedata/test_D.msh',grid_function=current_model.dirichlet_fun)
    export(f'./modedata/test_D_truth.msh',grid_function=dirichlet_fun)
    
    
    coeff = current_model.points_dirichlet(sphere.face_centers())
    print(coeff.shape)
    print(sphere.faces.shape)
    export(f'./modedata/sphere_predict.msh', grid_function=GridFunction(sphere.dp0_space, 
                                                            coefficients=coeff))
    coeff = []
    for point in sphere.face_centers():
        poles.reset(k,point)
        poles.dirichlet_reset()
        coeff.append((poles.dirichlet*weights).sum())
    coeff = np.asarray(coeff)
    dirichlet_fun =  GridFunction(sphere.dp0_space, coefficients=coeff)
    export(f'./modedata/shpere_truth.msh',grid_function=dirichlet_fun)

check()