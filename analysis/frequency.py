import numpy as np

def hz2mel(f):
    m = 2595*np.log10(1+f/700)
    return m

def mel2hz(m):
    f = (10**(m/2595)-1)*700
    return f

class FrequencyScale():
    def __init__(self, res, min_freq = 500, max_freq = 20000):
        self.resolution = res
        self.m_min = hz2mel(min_freq)
        self.m_max = hz2mel(max_freq)
        self.spacing = (self.m_max - self.m_min)/self.resolution

    def hz2index(self, f):
        m = hz2mel(f)
        return int((m-self.m_min)/self.spacing)

    def omega2index(self, omega):
        f = omega/(2*np.pi)
        return self.hz2index(f)

    def index2hz(self, i):
        m = self.m_min + (i+0.5)*self.spacing
        return mel2hz(m)

    def index2omega(self, i):
        f = self.index2hz(i)
        return f*(2*np.pi)

    def adjust_omega(self, omega):
        idx = self.omega2index(omega)
        return self.index2omega(idx)
        

    
