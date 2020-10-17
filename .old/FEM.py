import numpy as np
from scipy.sparse import csr_matrix,coo_matrix
from scipy.sparse.linalg import eigsh
import multiprocessing
import configparser
import os

class FEM_model():
    def __init__(self, vertices = None, tets = None):
        self.vertices = np.asarray(vertices)
        self.tets = np.asarray(tets)

    def set_material(self, mat):
        mat_config = configparser.ConfigParser()
        mat_config.read(f'analysis/material/material-{mat}.cfg')    
        m = mat_config['DEFAULT']
        self.youngs = float(m['youngs'])
        self.poison = float(m['poison'])
        self.alpha = float(m['alpha'])
        self.beta = float(m['beta'])
        self.density = float(m['density'])

    def compute_matrix(self, threadnum = 4):
        inputs = [[self, i, threadnum] for i in range(threadnum)]
        pool = multiprocessing.Pool(processes=threadnum)
        pool_outputs = pool.map(global_matrix_thread, inputs)
        pool.close()
        pool.join()
        result_m = pool_outputs[0][0]
        result_k = pool_outputs[0][1]
        for result in pool_outputs[1:]:
            result_m = np.concatenate((result_m,result[0]),axis=1)
            result_k = np.concatenate((result_k,result[1]),axis=1)
        size = len(self.vertices)*3
        self.mass_matrix = coo_matrix((result_m[0], (result_m[1], result_m[2])), shape=(size, size))
        self.stiff_matrix = coo_matrix((result_k[0], (result_k[1], result_k[2])), shape=(size, size))

    def compute_modes(self, min_freq = 20, max_freq = 20000, modes_num = 100):
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
    


def global_matrix_thread(params):
    mesh, r, threadnum = params
    Ms = []
    rows_m = []
    cols_m = []
    Ks = []
    rows_k = []
    cols_k = []
    size = len(mesh.tets)
    m = get_M(mesh.density)
    E = get_E(mesh.youngs,mesh.poison)
    for idx in range(size):
        if idx % threadnum == r:
            ids = mesh.tets[idx]
            rowid = np.vstack([ids*3,ids*3+1,ids*3+2]).T.reshape(-1)
            col = np.vstack([rowid]*12)
            row = col.T
            x,y,z = mesh.vertices[ids].T
            a,b,c,v = get_abcV(x,y,z)
            B = get_B(a,b,c,v)
            K = v*B.T.dot(E).dot(B).astype(np.float)
            M = m*v
            # K[:12,:6] = B.T.dot(E)
            # print(v)
            #K[:6,:12] = B
            args = (M != 0)
            Ms.append(M[args])
            rows_m.append(row[args])
            cols_m.append(col[args])
            args = (K != 0)
            Ks.append(K[args])
            rows_k.append(row[args])
            cols_k.append(col[args])
    data_m = np.hstack(Ms)
    row_m = np.hstack(rows_m)
    col_m = np.hstack(cols_m)
    data_k = np.hstack(Ks)
    row_k = np.hstack(rows_k)
    col_k = np.hstack(cols_k)
    return [[data_m,row_m,col_m],[data_k,row_k,col_k]]


def get_B(a,b,c,V):
    return np.array([
        [a[0],0   ,0   ,a[1],0   ,0   ,a[2],0   ,0   ,a[3],0   ,0   ],
        [0   ,b[0],0   ,0   ,b[1],0   ,0   ,b[2],0   ,0   ,b[3],0   ],
        [0   ,0   ,c[0],0   ,0   ,c[1],0   ,0   ,c[2],0   ,0   ,c[3]],
        [b[0],a[0],0   ,b[1],a[1],0   ,b[2],a[2],0   ,b[3],a[3],0   ],
        [0   ,c[0],b[0],0   ,c[1],b[1],0   ,c[2],b[2],0   ,c[3],b[3]],
        [c[0],0   ,a[0],c[1],0   ,a[1],c[2],0   ,a[2],c[3],0   ,a[3]]
    ])/(6*V)

def get_E(E0,v):
    return np.array([
        [1-v ,v   ,v   ,0    ,0    ,0    ],
        [v   ,1-v ,v   ,0    ,0    ,0    ],
        [v   ,v   ,1-v ,0    ,0    ,0    ],
        [0   ,0   ,0   ,0.5-v,0    ,0    ],
        [0   ,0   ,0   ,0    ,0.5-v,0    ],
        [0   ,0   ,0   ,0    ,0    ,0.5-v]
    ])*E0/(1+v)/(1-2*v)

def get_abcV(x,y,z):
    a = np.zeros(4)
    b = np.zeros(4)
    c = np.zeros(4)
    a[0]=y[1]*(z[3] - z[2])-y[2]*(z[3] - z[1])+y[3]*(z[2] - z[1])
    a[1]=-y[0]*(z[3] - z[2])+y[2]*(z[3] - z[0])-y[3]*(z[2] - z[0])
    a[2]=y[0]*(z[3] - z[1])-y[1]*(z[3] - z[0])+y[3]*(z[1] - z[0])
    a[3]=-y[0]*(z[2] - z[1])+y[1]*(z[2] - z[0])-y[2]*(z[1] - z[0])
    b[0]=-x[1]*(z[3] - z[2])+x[2]*(z[3] - z[1])-x[3]*(z[2] - z[1])
    b[1]=x[0]*(z[3] - z[2])-x[2]*(z[3] - z[0])+x[3]*(z[2] - z[0])
    b[2]=-x[0]*(z[3] - z[1])+x[1]*(z[3] - z[0])-x[3]*(z[1] - z[0])
    b[3]=x[0]*(z[2] - z[1])-x[1]*(z[2] - z[0])+x[2]*(z[1] - z[0])
    c[0]=x[1]*(y[3] - y[2])-x[2]*(y[3] - y[1])+x[3]*(y[2] - y[1])
    c[1]=-x[0]*(y[3] - y[2])+x[2]*(y[3] - y[0])-x[3]*(y[2] - y[0])
    c[2]=x[0]*(y[3] - y[1])-x[1]*(y[3] - y[0])+x[3]*(y[1] - y[0])
    c[3]=-x[0]*(y[2] - y[1])+x[1]*(y[2] - y[0])-x[2]*(y[1] - y[0])
    V=((x[1] - x[0])*((y[2] - y[0])*(z[3] - z[0])-(y[3] - y[0])*(z[2] - z[0]))+(y[1] - y[0])*((x[3] - x[0])*(z[2] - z[0])-(x[2] - x[0])*(z[3] - z[0]))+(z[1] - z[0])*((x[2] - x[0])*(y[3] - y[0])-(x[3] - x[0])*(y[2] - y[0])))/6
    return a,b,c,abs(V)
    
def get_Me(d, v):
    return get_M(d)*v

def get_M(d):
    m =  np.array([
                    [1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
                ],dtype=np.float)
    return (m + m.T)*d/20
