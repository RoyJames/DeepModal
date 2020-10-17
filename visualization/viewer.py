import numpy as np
import os
from glumpy import app, gl, gloo, data, log, glm
from glumpy.transforms import Trackball, Position
app.use("qt5")
import sys

def load_shader(path):
    with open(path + '.vert', 'r') as f:
        vertex = f.read()
    with open(path + '.frag', 'r') as f:
        fragment = f.read()
    return gloo.Program(vertex, fragment)

class Viewer(object):
    def __init__(self, width = 500, height = 500):
        self.mesh_shader = load_shader('visualization/shader/mesh')
        trackball = Trackball(Position("position"))
        self.mesh_shader['transform'] = trackball
        trackball.theta, trackball.phi, trackball.zoom = 80, -135, 5
        self.img_shader = load_shader('visualization/shader/image')
        self.img_shader['position']= [(-1,-1), (-1,1), (1,-1), (1,1)]
        self.window = self.set_window(width, height)
        self.click = None
        self.click_callback = None
        self.key_dict = {}
        self.vertices = None

    def run(self):
        app.run()
        
    def connect_click(self, callback):
        '''
        Parameters
        ----------
        callback : function(click_face_index)
        '''
        self.click_callback = callback

    def connect_key(self, key, callback):
        '''
        Parameters
        ----------
        key : string
        callback : function() for key pressed
        '''
        self.key_dict[key] = callback

    def load_mesh(self, vertices, faces, normals):
        vertices,faces,normals = map(np.asarray, [vertices, faces, normals])
        vertices = vertices / 2
        self.vertices = vertices
        self.faces = faces
        self.normals = normals
        V = np.zeros(len(faces)*3, [("position", np.float32, 3),
                                        ("normal", np.float32, 3),
                                        ("id", np.float32, 1)])
        V['position'] = vertices[faces].reshape(-1,3)
        V['normal'] = normals[faces].reshape(-1,3)
        V['id'] = (np.arange(0,len(faces)*3)//3 + 1)
        V = V.view(gloo.VertexBuffer)
        self.mesh_shader.bind(V)
        self.mesh_shader['select_id'] = -1
        self.load_color(np.ones_like(normals)*0.5)

    def load_color(self, colors):
        V = np.zeros(len(self.faces)*3, [("color", np.float32, 3)])
        colors = np.asarray(colors)
        # print(colors.shape, self.faces.shape)
        V['color'] = colors[self.faces].reshape(-1,3)
        V = V.view(gloo.VertexBuffer)
        self.mesh_shader['color'] = V

    def render(self):
        self.img_shader.draw(gl.GL_TRIANGLE_STRIP)

    def update(self):
        model = self.mesh_shader['transform']['model'].reshape(4,4)
        view  = self.mesh_shader['transform']['view'].reshape(4,4)
        self.mesh_shader['m_view']  = view
        self.mesh_shader['m_model'] = model
        self.mesh_shader['m_normal'] = np.array(np.matrix(np.dot(view, model)).I.T)

    def set_framebuffer(self, width, height):
        color = np.zeros((height,width,4),np.uint8).view(gloo.Texture2D)
        color.interpolation = gl.GL_LINEAR
        pick = np.zeros((height,width,4),np.uint8).view(gloo.Texture2D)
        pick.interpolation = gl.GL_LINEAR
        framebuffer = gloo.FrameBuffer(color=[color,pick], depth=gloo.DepthBuffer(width, height))
        self.framebuffer = framebuffer
        self.img_shader["color"] = self.framebuffer.color[0]
        
    def set_window(self, width, height):
        window = app.Window(width=width, height=height, color=(0.6,0.6,0.6,1))
        self.width = width
        self.height = height
        self.set_framebuffer(window.width,window.height)
        @window.event
        def on_draw(dt):
            if self.vertices is None:
                return 
            gl.glEnable(gl.GL_DEPTH_TEST)
            self.framebuffer.activate()
            window.clear()
            self.mesh_shader.draw(gl.GL_TRIANGLES)
            if self.click is not None:
                gl.glReadBuffer(gl.GL_COLOR_ATTACHMENT1)
                r,g,b,a = gl.glReadPixels(self.click[0],self.click[1],1,1, gl.GL_RGBA, gl.GL_UNSIGNED_BYTE)
                if type(r) is not int: r = ord(r)
                if type(g) is not int: g = ord(g)
                if type(b) is not int: b = ord(b)
                index = b + 256*g + 256*256*r
                self.mesh_shader['select_id'] = index
                self.click = None
                if self.click_callback is None:
                    print('Click callback function not defined')
                else:
                    self.click_callback(index - 1)
            self.framebuffer.deactivate()
            window.clear()
            self.render() 
        
        @window.event
        def on_init():
            gl.glEnable(gl.GL_DEPTH_TEST)
            self.update()
        
        @window.event
        def on_mouse_drag(x, y, dx, dy, button):
            self.update()
            
        @window.event
        def on_resize(width, height):
            self.width = width
            self.height = height
            self.set_framebuffer(width, height)

        @window.event
        def on_mouse_press(x, y, button):
            if (button == 8): #right click
                self.click = int(x), self.height-int(y)
        
        @window.event
        def on_key_press(symbol, modifiers):
            keys = self.key_dict.keys()
            if len(keys) == 0:
                print(f'key pressed: {symbol}')
            elif symbol in keys:
                self.key_dict[symbol]()

        window.attach(self.mesh_shader['transform'])
        return window

