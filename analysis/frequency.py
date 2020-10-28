import numpy as np

def hz2mel(f):
    m = 2595*np.log10(1+f/700)
    return m

def mel2hz(m):
    f = (10**(m/2595)-1)*700
    return f

class FrequencyScale():
    def __init__(self, res = 32, min_freq = 100, max_freq = 10000):
        self.resolution = res
        self.m_min = hz2mel(min_freq)
        self.m_max = hz2mel(max_freq)
        self.spacing = (self.m_max - self.m_min)/self.resolution

    def hz2index(self, f):
        m = hz2mel(f)
        return int((m-self.m_min)/self.spacing)

    def index2hz(self, i):
        m = self.m_min + (i+0.5)*self.spacing
        return mel2hz(m)

    def adjust_omega(self, omega):
        f = omega / (2*np.pi)
        idx = self.hz2index(f)
        f = self.index2hz(idx)
        return f*(2*np.pi)
