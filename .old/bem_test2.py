
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

    def normals(self):
        return self.grid.normals

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
        self.left_side = self.hyper_single
        self.right_side = (-0.5*self.identity-self.adjoint_double)*self.neumann_fun
        dirichlet_fun, info, _ = bempp.api.linalg.gmres(self.left_side, self.right_side,tol=1e-6, return_residuals=True)
        self.dirichlet_fun = dirichlet_fun
        export(f'./modedata/left.msh', grid_function = self.left_side*dirichlet_fun)
        export(f'./modedata/right.msh', grid_function = self.right_side)

    def points_dirichlet(self, points):
        potential_single = potential.helmholtz.single_layer(self.dp0_space, points.T, self.k)
        potential_double = potential.helmholtz.double_layer(self.p1_space, points.T, self.k)
        dirichlet = -potential_single*self.neumann_fun + potential_double*self.dirichlet_fun 
        return dirichlet.reshape(-1)

def check():
    vox = Hexa_model('dataset/7.obj')
    vox.create_tetra_and_boundary()
    vox.set_transform([center_scale(0.2)])
    print(vox.tets.shape)
    return
    r_in = 0.1
    r_out = 0.2
    sphere = boundary_mesh(grid=bempp.api.shapes.sphere(h=0.02,r = r_out))
    current_model = boundary_mesh(vertices=vox.vertices, faces=vox.boundary_faces)
    c = 343
    omega = 1000*6.28
    k = omega / c
    print(k)
    scale = 10
    poles = special.Multipole(scale)
    weights = np.zeros(poles.pole_number)
    weights[1] = 0.3
    weights[5] = 0.3
    neumann_coeff = []
    dirichlet_coeff = []
    for point,normal in zip(current_model.face_centers(), current_model.normals()):
        #print(point, normal)
        poles.reset(k,point)
        poles.dirichlet_reset()
        poles.neumann_reset(normal)
        neumann_coeff.append((poles.neumann*weights).sum())
        dirichlet_coeff.append((poles.dirichlet*weights).sum())

    neumann_fun =  GridFunction(current_model.dp0_space, coefficients=np.asarray(neumann_coeff))
    dirichlet_fun = GridFunction(current_model.dp0_space, coefficients=np.asarray(dirichlet_coeff))

    os.makedirs(f'./modedata',exist_ok=True)
    current_model.set_wave_number(k)
    current_model.set_neumann_fun(neumann_fun)
    current_model.ext_neumann2dirichlet()
    
    export(f'./modedata/test_N.msh',grid_function=current_model.neumann_fun)
    
    identity = boundary.sparse.identity(
                        current_model.p1_space, current_model.dp0_space, current_model.p1_space)
    export(f'./modedata/test_D.msh',grid_function=identity*current_model.dirichlet_fun)
    export(f'./modedata/test_D_truth.msh',grid_function=dirichlet_fun)

    

    
    coeff = current_model.points_dirichlet(sphere.face_centers())
    print(coeff.shape)
    print(sphere.faces.shape)
    export(f'./modedata/sphere_predict.msh', grid_function=GridFunction(sphere.dp0_space, 
                                                            coefficients=coeff))
    coeff = []
     

check()
