import bempp.api
from bempp.api.operators import potential, boundary
from bempp.api import GridFunction, export
import numpy as np
from analysis import frequency, special

bempp.api.LOGGER.propagate = False
SPEED_OF_SOUND = 343
AIR_DENSITY = 1.225

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
        self.cache_k = -1
    
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
        if self.cache_k != self.k:
            self.adjoint_double = boundary.helmholtz.adjoint_double_layer(
                            self.dp0_space, self.dp0_space, self.p1_space, self.k, precision="single", device_interface='opencl')
            self.hyper_single = boundary.helmholtz.hypersingular(
                            self.p1_space, self.dp0_space, self.p1_space, self.k, precision="single", device_interface='opencl')
            self.identity = boundary.sparse.identity(
                            self.dp0_space, self.dp0_space, self.p1_space, precision="single")
            self.cache_k = self.k
        
        self.left_side = self.hyper_single
        self.right_side = (-0.5*self.identity-self.adjoint_double)*self.neumann_fun
        dirichlet_fun, info, _ = bempp.api.linalg.gmres(self.left_side, self.right_side, tol=1e-5, maxiter=1000, return_residuals=True)
        # export('left.msh', grid_function= self.left_side*dirichlet_fun)
        # export('right.msh', grid_function= self.right_side)
        self.dirichlet_fun = dirichlet_fun

    def points_dirichlet(self, points):
        potential_single = potential.helmholtz.single_layer(self.dp0_space, points.T, self.k)
        potential_double = potential.helmholtz.double_layer(self.p1_space, points.T, self.k)
        dirichlet = -potential_single*self.neumann_fun + potential_double*self.dirichlet_fun 
        return dirichlet.reshape(-1)



class PolesMatrix():
    def __init__(self, freq_res = 32, poles_scale = 10):
        self.poles = special.Multipole(poles_scale)
        self.freq_scale = frequency.FrequencyScale(freq_res)
        self.wave_numbers = self.freq_scale.index2omega(np.arange(freq_res)) / SPEED_OF_SOUND

    def wavenumber2index(self, k):
        omega = k*SPEED_OF_SOUND
        return self.freq_scale.omega2index(omega)

    def sample_points(self, h):
        r_unit = 0.1*3**0.5
        r1 = 1.6*r_unit
        r2 = 2.6*r_unit
        r3 = 3.4*r_unit
        sphere1 = boundary_mesh(grid=bempp.api.shapes.sphere(r=r1,h=h))
        sphere2 = boundary_mesh(grid=bempp.api.shapes.sphere(r=r2,h=h))
        sphere3 = boundary_mesh(grid=bempp.api.shapes.sphere(r=r3,h=h))
        self.points = np.concatenate([sphere1.face_centers(), sphere2.face_centers(), sphere3.face_centers()])
        # print('shape of sampling points:')
        # print(self.points.shape)
    
    def assemble_matrix(self):
        self.all_matrix = []
        for k in self.wave_numbers:
            dirichlet_matrix = []
            for p in self.points:
                self.poles.reset(k,x=p)
                self.poles.dirichlet_reset()
                dirichlet_matrix.append(self.poles.dirichlet)
            self.all_matrix.append(dirichlet_matrix)
        self.all_matrix = np.asarray(self.all_matrix)
        # print('shape of all matrix')
        # print(self.all_matrix.shape)

    def get_grid_function(self, sphere_mesh, weights, k):
<<<<<<< HEAD
        coeff = self.get_sphere_coeff(sphere_mesh, weights, k)
        dirichlet_fun =  GridFunction(sphere_mesh.dp0_space, coefficients=coeff)
        return dirichlet_fun

    def get_grid_fun_from_list(self, sphere_mesh, weights_, k):
        coeff = np.zeros(len(sphere_mesh.face_centers()),dtype=np.complex)
        for weights in weights_:
            coeff += (self.get_sphere_coeff(sphere_mesh, weights, k))**2
        coeff /= len(weights_)
        dirichlet_fun =  GridFunction(sphere_mesh.dp0_space, coefficients=coeff**0.5)
        return dirichlet_fun

    def get_sphere_coeff(self, sphere_mesh, weights, k):
=======
>>>>>>> 9e91e9052ddc2f40996d02a5b6d3292290e83072
        coeff = []
        for p in sphere_mesh.face_centers():
            self.poles.reset(k, p)
            self.poles.dirichlet_reset()
            coeff.append((self.poles.dirichlet*weights).sum())
        coeff = np.asarray(coeff)
<<<<<<< HEAD
        return coeff
=======
        dirichlet_fun =  GridFunction(sphere_mesh.dp0_space, coefficients=coeff)
        return dirichlet_fun
>>>>>>> 9e91e9052ddc2f40996d02a5b6d3292290e83072
