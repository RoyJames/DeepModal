import numpy as np
import configparser
from scipy.sparse.linalg import eigsh
from scipy.sparse import csr_matrix,coo_matrix
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
mod = SourceModule(open('analysis/cuda/compute_matrix.cu').read())
cuda_stiff_matrix_computer = mod.get_function("compute_stiff_matrix")
cuda_mass_matrix_computer = mod.get_function("compute_mass_matrix")
# compute_matrix(float *values, int* rows, int* cols, int res, float youngs, float poison, float* vertices, int *tets, int tets_num)

class Material:
    Ceramic = 2700,7.2E10,0.19,6,1E-7
    Glass = 2600,6.2E10,0.20,1,1E-7
    Wood = 750,1.1E10,0.25,60,2E-6
    Plastic = 1070,1.4E9,0.35,30,1E-6
    Iron = 8000,2.1E11,0.28,5,1E-7
    Polycarbonate = 1190,2.4E9,0.37,0.5,4E-7
    Steel = 7850,2.0E11,0.29,5,3E-8

class FEM_model():
    def __init__(self, vertices, tets):
        self.vertices = np.asarray(vertices).reshape(-1,3)
        self.tets = np.asarray(tets).reshape(-1,4)

    def set_material(self, mat):
        self.density, self.youngs, self.poison, self.beta, self.alpha = mat
        
    def create_matrix(self):
        num = self.tets.shape[0]*12*12
        cuda_res = 64
        values = np.zeros(num,dtype = np.float64)
        rows = np.zeros(num,dtype = np.int32)
        cols = np.zeros(num,dtype = np.int32)
        cuda_stiff_matrix_computer(
            drv.Out(values), drv.Out(rows), drv.Out(cols),
            np.int32(cuda_res), np.float64(self.youngs), np.float64(self.poison),
            drv.In(self.vertices.astype(np.float64).reshape(-1)),
            drv.In(self.tets.astype(np.int32).reshape(-1)),
            np.int32(self.tets.shape[0]),
            block=(cuda_res,1,1), grid=(cuda_res,cuda_res)
        )
        size = len(self.vertices)*3
        self.stiff_matrix = coo_matrix((values, (rows, cols)), shape=(size, size))
        self.stiff_matrix.eliminate_zeros()
        #self.stiff_matrix.sum_duplicates()

        values = np.zeros(num,dtype = np.float64)
        rows = np.zeros(num,dtype = np.int32)
        cols = np.zeros(num,dtype = np.int32)
        cuda_mass_matrix_computer(
            drv.Out(values), drv.Out(rows), drv.Out(cols),
            np.int32(cuda_res), np.float64(self.density),
            drv.In(self.vertices.astype(np.float64).reshape(-1)),
            drv.In(self.tets.astype(np.int32).reshape(-1)),
            np.int32(self.tets.shape[0]),
            block=(cuda_res,1,1), grid=(cuda_res,cuda_res)
        )
        self.mass_matrix = coo_matrix((values, (rows, cols)), shape=(size, size))
        self.mass_matrix.eliminate_zeros()
        #self.mass_matrix.sum_duplicates()

    def compute_modes(self, min_freq = 20, max_freq = 20000, modes_num = 30):
        sigma = ((2*np.pi*max_freq)**2 + (2*np.pi*min_freq)**2)/2
        vals, vecs = eigsh(self.stiff_matrix, k=modes_num, M=self.mass_matrix,which='LM',sigma=sigma)
        while max(vals) < (2*np.pi*max_freq)**2 :
            modes_num += 100
            vals, vecs = eigsh(self.stiff_matrix, k=modes_num, M=self.mass_matrix,which='LM',sigma=sigma)
        valid = (vals > (min_freq*2*np.pi)**2)&(
                vals < (max_freq*2*np.pi)**2)
        # print(vals[:100])
        self.vals = vals[valid]
        self.vecs = vecs[:,valid]
        self.c = (self.alpha*self.vals + self.beta)
        self.omega = np.sqrt(self.vals)
        self.omega_d = self.omega*np.sqrt(1 - self.c**2/(self.omega**2*4))
    
