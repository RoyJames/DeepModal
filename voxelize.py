
import bempp.api
from bempp.api import export, GridFunction
from analysis.voxelize import Hexa_model, center_scale
from analysis.FEM import FEM_model
from analysis.BEM import boundary_mesh, PolesMatrix, SPEED_OF_SOUND, AIR_DENSITY
import numpy as np
import os
from scipy.linalg import lstsq
from tqdm import tqdm
from glob import glob
import shutil
def work(file_list, root):
    for filename in tqdm(file_list):
        #=========================voxelize=====================
        vox = Hexa_model(filename)
        vox.create_tetra_and_boundary()
        dirname = root + os.path.basename(os.path.dirname(filename)) + '/' + os.path.basename(filename)[:-4]
        if len(vox.tets) > 20000 or len(vox.boundary_faces) > 12000:
            if os.path.exists(dirname):
                shutil.rmtree(dirname)
            continue

        vox.set_transform([center_scale(0.15)])
        os.makedirs(dirname, exist_ok=True)
        if os.path.exists(dirname+'/tets.npy'):
            old_num = len(np.load(dirname+'/tets.npy'))
        else:
            old_num = 0

        if old_num != len(vox.tets):
            print(filename)
            if os.path.exists(dirname+'/displacements.npy'):
                os.remove(dirname+'/displacements.npy') 
            np.save(dirname+'/vertices', vox.vertices)
            np.save(dirname+'/boundary_faces', vox.boundary_faces)
            np.save(dirname+'/tets', vox.tets)
       
import sys
if __name__ == "__main__":
    lst = ['bottle','bowl','cone','cup','flower_pot','glass_box','radio','vase','xbox']
    file_list = []
    for category in lst:
        file_list += glob(f'/data1/modelnet40/{category}/*/*.off')
    num = 300
    step = 7
    for i in range(step):
        work(file_list[i*num:(i+1)*num], f'dataset/{i}/')