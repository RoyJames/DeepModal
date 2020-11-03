
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

def work(file_list, root):
    for filename in tqdm(file_list):
        #=========================voxelize=====================
        vox = Hexa_model(filename)
        vox.create_tetra_and_boundary()
        if len(vox.tets) > 20000:
            continue
        if len(vox.boundary_faces) > 12000:
            continue
        vox.set_transform([center_scale(0.15)])
        dirname = root + os.path.basename(os.path.dirname(filename))
        os.makedirs(dirname, exist_ok=True)
        np.save(dirname+'/vertices', vox.vertices)
        np.save(dirname+'/boundary_faces', vox.boundary_faces)
        np.save(dirname+'/tets', vox.tets)
        
import sys
if __name__ == "__main__":
    num = 1000
    step = 10
    for i in range(step):
        file_list = glob('../../abcDataset/abc/*/*.obj')[i*num:(i+1)*num]
        print('file list gotten')
        work(file_list, f'dataset/{i}/')