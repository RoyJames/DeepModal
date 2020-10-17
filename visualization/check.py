from .viewer import Viewer
from PyQt5.QtWidgets import QMainWindow, QWidget, QVBoxLayout, QPushButton, QGridLayout, QLabel, QSlider, QFileDialog
from PyQt5.QtCore import Qt
from glumpy import app
import open3d as o3d
import numpy as np

def run():
    check().run()
    
class check():
    def __init__(self,home_dir = 'G:/dataset/test/', modes_name1 = '/vecs_norm.npy', modes_name2 = '/vecs_norm_.npy'):
        self.home_dir = home_dir
        self.modes_name1 = modes_name1
        self.modes_name2 = modes_name2
        self.dir = None
        self.modes1 = self.modes2 = None

        v1 = Viewer()
        v2 = Viewer()
        v1.window._native_window.setGeometry(400,200,500,500)
        v2.window._native_window.setGeometry(900,200,500,500)
        self.v1 = v1
        self.v2 = v2

        class MMainWindow(QMainWindow):
            def __init__(self):
                super().__init__()

            def closeEvent(self, event):
                v1.window.close()
                v2.window.close()
                super().closeEvent(event)

        widget = QWidget()
        
        layout = QVBoxLayout()
        load_button = QPushButton('load...')
        load_button.clicked.connect(self.load)
        load_modes_button = QPushButton('load modes')
        load_modes_button.clicked.connect(self.load_modes)
        self.sld = QSlider(Qt.Vertical)
        self.sld.valueChanged[int].connect(self.show_mode)
        
        layout.addWidget(load_button)
        layout.addWidget(load_modes_button)
        layout.addWidget(self.sld)
        widget.setLayout(layout)

        main_window = MMainWindow()
        main_window.setCentralWidget(widget)
        main_window.setGeometry(200,100,200,800)
        main_window.show()
        self.main_window = main_window
    
    def run(self):
        app.run()

    def hsl_to_rgb(self, h):
        h = h / (np.abs(h.max()) + 1e-10)
        h = (1 - h)*2/3
        s = np.ones_like(h)
        l = np.ones_like(h)*0.3
        def hue_to_rgb(p, q, t):
            t = np.clip(t,0,1)
            ret = p.copy()
            ret[t < 2/3] = (p + (q - p) * (2/3 - t) * 6)[t < 2/3]
            ret[t < 1/2] = q[t < 1/2]
            ret[t < 1/6] = (p + (q - p) * 6 * t)[t < 1/6]
            return ret.reshape(-1,1)
        q = np.zeros_like(l)
        q[l < 0.5] = (l * (1 + s))[l < 0.5]
        q[l >= 0.5] = (l + s - l * s)[l >= 0.5]
        p = 2 * l - q
        r = hue_to_rgb(p, q, h + 1/3)
        g = hue_to_rgb(p, q, h)
        b = hue_to_rgb(p, q, h - 1/3)
        return np.concatenate([r, g, b],axis=1)

    def load(self):
        self.dir = QFileDialog.getExistingDirectory(self.main_window, 'Open directory', self.home_dir)
        self.load_mesh()

    def load_mesh(self):
        mesh = o3d.io.read_triangle_mesh(self.dir + '/mesh.ply')
        self.v1.load_mesh(mesh.vertices, mesh.triangles, mesh.vertex_normals)
        self.v2.load_mesh(mesh.vertices, mesh.triangles, mesh.vertex_normals)
        self.modes1 = self.modes2 = None

    def load_modes(self):
        if self.dir is None:
            print('mesh not loaded')
            return
        self.modes1 =  np.load(self.dir + self.modes_name1)
        self.modes2 =  np.load(self.dir + self.modes_name2)
        self.modes1 = self.modes1.reshape(-1,3,self.modes1.shape[-1])
        self.modes2 = self.modes2.reshape(-1,3,self.modes2.shape[-1])
        self.modes1 = (self.modes1**2).sum(1)
        self.modes2 = (self.modes2**2).sum(1)
        self.sld.setMaximum(self.modes1.shape[-1] - 1)
        self.sld.setValue(0)
        self.show_mode(0)
        
    def show_mode(self, idx):
        if self.modes1 is None or self.modes2 is None:
            print('modes not loaded')
            return
        mode1 = self.modes1[...,idx]
        mode2 = self.modes2[...,idx]
        data1 = self.hsl_to_rgb(mode1)
        data2 = self.hsl_to_rgb(mode2)
        self.v1.load_color(data1)
        self.v2.load_color(data2)

    def window_widget(self, window):
        w = QWidget()
        w.setLayout(QVBoxLayout())
        w.layout().addWidget(window._native_window)
        return w
    
    
    

    

    