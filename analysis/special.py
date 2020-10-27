import numpy as np

class Multipole():
    def __init__(self, scale = 10):
        self.scale = 10
        n_max = self.scale + 2
        self.n_max = n_max
        self.h = np.zeros(n_max).astype(np.complex)
        self.p = np.zeros((n_max, n_max)).astype(np.complex)
        self.coeff = np.zeros((n_max, n_max)).astype(np.complex)

    def reset(self, k, x=None, theta=None, phi=None, r=None):
        if x is not None:
            r = (x[0]**2+x[1]**2+x[2]**2)**0.5
            phi = np.arctan2(x[1],x[0])
            theta =  np.arccos(x[2]/r)
        self.k = k
        self.r = r
        self.phi = phi
        self.theta = theta
        #=======================coefficent========================
        self.coeff[0,:] = (2*np.arange(self.n_max)+1)/(4*np.pi)
        for m in range(1,self.n_max):
            up = np.arange(self.n_max) - (m-1)
            up[up <= 0] = 1
            down = np.arange(self.n_max) + m
            self.coeff[m,:] = self.coeff[m-1,:]/(up*down)
        #=================hankel function of first kind================
        z = k*r
        self.h[0] = -1j/z*np.exp(1j*z)
        self.h[1] = -(z+1j)/z**2*np.exp(1j*z)
        for i in range(2,self.n_max):
            self.h[i] = (2*i-1)/z*self.h[i-1]-self.h[i-2] 

        #================Associated Laguerre Polynomial===============
        self.p[0][0] = 1
        for i in range(self.n_max-1):
            self.p[i+1,i+1] =  -(2*i+1)*(1-np.cos(phi)**2)**0.5*self.p[i,i]
        for m in range(self.n_max - 1):
            self.p[m,m+1] = np.cos(phi)*(2*m+1)*self.p[m,m]
            for i in range(m+1, self.n_max - 1):
                self.p[m,i+1] = ((2*i+1)*np.cos(phi)*self.p[m,i]-(i+m)*self.p[m,i-1])/(i-m+1)
    
    def sph_harm(self, m, n):
        Y = self.coeff[abs(m),n]**0.5*self.p[abs(m),n]*np.exp(1j*abs(m)*self.theta)
        if m < 0:
            Y = (-1)**m*Y.conjugate()
        return Y
    
    def hankel(self, n, derivative = False):
        if derivative==False:
            return self.h[n]
        else:
            return self.k*(-self.h[n+1]+n/(self.k*self.r)*self.h[n])

    def evaluate(self, m, n, neumann = False):
        return self.hankel(n,neumann)*self.sph_harm(m,n)

    def neumann_reset(self):
        self.neumann = []
        for n in range(self.scale):
            for m in range(-n,n+1):
                self.neumann.append(self.evaluate(m,n,True))
        self.neumann = np.asarray(self.neumann)

    def dirichlet_reset(self):
        self.dirichlet = []
        for n in range(self.scale):
            for m in range(-n,n+1):
                self.dirichlet.append(self.evaluate(m,n,False))
        self.dirichlet = np.asarray(self.dirichlet)
        

