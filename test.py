
# import bempp.api
# from analysis.cpp import boundary
# from analysis.voxelize import VOX
# import numpy as np
# import open3d as o3d
# import analysis.FEM as FEM

# vox = VOX('dataset/4.obj')
# vox.create_tetra_mesh()
# vox.create_boundary()

# grid = bempp.api.Grid(vox.vertices.transpose().astype(np.float64), 
#                     vox.boundary_faces.transpose().astype(np.uint32))


# grid.plot()

import bempp.api
import numpy as np
grid = bempp.api.shapes.sphere(h=0.1)
dp0_space = bempp.api.function_space(grid, "DP", 0)
p1_space = bempp.api.function_space(grid, "P", 1)
identity = bempp.api.operators.boundary.sparse.identity(
    p1_space, p1_space, dp0_space)
dlp = bempp.api.operators.boundary.laplace.double_layer(
    p1_space, p1_space, dp0_space)
slp = bempp.api.operators.boundary.laplace.single_layer(
    dp0_space, p1_space, dp0_space)
@bempp.api.real_callable
def dirichlet_data(x, n, domain_index, result):
    result[0] = 1./(4 * np.pi * ((x[0] - .9)**2 + x[1]**2 + x[2]**2)**(0.5))
    
dirichlet_fun = bempp.api.GridFunction(p1_space, fun=dirichlet_data)
dirichlet_fun.plot()



