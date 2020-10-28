import bempp.api
import numpy as np	 
from scipy import special
from bempp.api import export

grid = bempp.api.shapes.sphere(h=0.1)	
dp0_space = bempp.api.function_space(grid, "DP", 0)	
p1_space = bempp.api.function_space(grid, "P", 1)	

n = 2
m = 2
k = 20

identity = bempp.api.operators.boundary.sparse.identity(	
    p1_space, p1_space, p1_space)	
dlp = bempp.api.operators.boundary.helmholtz.double_layer(	
    p1_space, p1_space, p1_space,k)	
slp = bempp.api.operators.boundary.helmholtz.single_layer(	
    dp0_space, p1_space, p1_space,k)

def hankel(n,z,derivative=False):
    return special.spherical_jn(n,z,derivative) + 1j*special.spherical_yn(n,z,derivative)

@bempp.api.complex_callable(jit=False)
def dirichlet_data(x, n_, domain_index, result):	
    r = (x[0]**2+x[1]**2+x[2]**2)**0.5
    #phi = np.arccos(x[0]/r)
    phi = np.arctan2(x[1],x[0])
    theta =  np.arccos(x[2]/r)
    result[0] = hankel(n,r*k)*special.sph_harm(m,n,phi,theta)

@bempp.api.complex_callable(jit=False)
def neumann_data(x, n_, domain_index, result):	
    r = (x[0]**2+x[1]**2+x[2]**2)**0.5
    # phi = np.arccos(x[0]/r)
    phi = np.arctan2(x[1],x[0])
    theta =  np.arccos(x[2]/r)
    result[0] = k*hankel(n,r*k, True)*special.sph_harm(m,n,phi,theta)

dirichlet_fun = bempp.api.GridFunction(p1_space, fun=dirichlet_data)
neumann_fun = bempp.api.GridFunction(dp0_space, fun=neumann_data)
export('neumann.msh',grid_function= neumann_fun)
export('dirichlet.msh', grid_function= dirichlet_fun)

dirichlet_fun_predict, info = bempp.api.gmres(-0.5*identity + dlp, slp*neumann_fun)
export('dirichlet_predict.msh', grid_function= dirichlet_fun_predict)