import numpy as np
from numba import jit

@jit(nopython=True)
def get_coeeff(coeff,n_max):
    coeff[0,:] = (2*np.arange(n_max)+1)/(4*np.pi)
    for m in range(1, n_max):
        up = np.arange( n_max) - (m-1)
        up[up <= 0] = 1
        down = np.arange(n_max) + m
        coeff[m,:] = coeff[m-1,:]/(up*down)
    return coeff

@jit(nopython=True)
def get_handel(h,z,n_max):
    h[0] = -1j/z*np.exp(1j*z)
    h[1] = -(z+1j)/z**2*np.exp(1j*z)
    for i in range(2,n_max):
        h[i] = (2*i-1)/z*h[i-1]-h[i-2]
    return h

@jit(nopython=True)
def get_alp(p, p_d, n_max, theta):
    p[0][0] = 1
    p_d[0][0] = 0
    for i in range(n_max-1):
        p[i+1,i+1] =  -(2*i+1)*(1-np.cos(theta)**2)**0.5*p[i,i]
        p_d[i+1,i+1] =  (2*i+1)*np.cos(theta)/(1-np.cos(theta)**2)**0.5*p[i,i] - (2*i+1)*(1-np.cos(theta)**2)**0.5*p_d[i,i]
        
    for m in range(n_max - 1):
        p[m,m+1] = np.cos(theta)*(2*m+1)*p[m,m]
        p_d[m,m+1] = (2*m+1)*p[m,m] + np.cos(theta)*(2*m+1)*p_d[m,m]
        for i in range(m+1, n_max - 1):
            p[m,i+1] = ((2*i+1)*np.cos(theta)*p[m,i]-(i+m)*p[m,i-1])/(i-m+1)
            p_d[m,i+1] = ((2*i+1)*p[m,i]+(2*i+1)*np.cos(theta)*p_d[m,i]-(i+m)*p_d[m,i-1])/(i-m+1)
    return p,p_d

class Multipole():
    def __init__(self, scale = 10):
        self.scale = 10
        n_max = self.scale + 2
        self.n_max = n_max
        self.h = np.zeros(n_max).astype(np.complex)
        self.p = np.zeros((n_max, n_max)).astype(np.complex)
        self.p_d = np.zeros((n_max, n_max)).astype(np.complex)
        self.coeff = np.zeros((n_max, n_max)).astype(np.complex)

        self.pole_number = 0
        for n in range(self.scale):
            for m in range(-n,n+1):
                self.pole_number += 1

    def reset(self, k, x=None, theta=None, phi=None, r=None):
        if x is not None:
            r = (x[0]**2+x[1]**2+x[2]**2)**0.5
            phi = np.arctan2(x[1],x[0])
            theta =  np.arccos(x[2]/r)
        self.k = k
        self.r = r
        self.phi = phi
        self.theta = theta
        n_max = self.n_max
        #=======================coefficent========================
        self.coeff = get_coeeff(self.coeff,n_max)
        #=================hankel function of first kind================
        z = k*r
        self.h = get_handel(self.h,z,n_max)
        #================Associated Laguerre Polynomial===============
        self.p, self.p_d = get_alp(self.p, self.p_d, n_max, theta)
        #=================== inverse Jacobian ==========================
        x,y,z = x[0],x[1],x[2]
        t1 = r*r*(x**2+y**2)**0.5
        t2 = (x**2+y**2)
        self.j_inv = np.array([
            [x/r,       y/r,        z/r     ],
            [x*z/t1,   y*z/t1,     -t2/t1  ],
            [-y/t2,     x/t2,       0       ]
        ])

    def sph_harm(self, m, n):
        Y = self.coeff[abs(m),n]**0.5*self.p[abs(m),n]*np.exp(1j*abs(m)*self.phi)
        if m < 0:
            Y = (-1)**m*Y.conjugate()
        return Y

    def d_sph_harm(self, m, n, d_theta, d_phi):
        d_Y = self.coeff[abs(m),n]**0.5*self.p_d[abs(m),n]*(-np.sin(self.theta)*d_theta)*np.exp(1j*abs(m)*self.phi) + \
            self.coeff[abs(m),n]**0.5*self.p[abs(m),n]*1j*abs(m)*d_phi*np.exp(1j*abs(m)*self.phi)
        if m < 0:
            d_Y = (-1)**m*d_Y.conjugate()
        return d_Y

    def hankel(self, n):
        return self.h[n]
    
    def d_hankel(self, n, d_r):
        return (-self.h[n+1]+n/(self.k*self.r)*self.h[n])*self.k*d_r

    def evaluate(self, m, n):
        return self.hankel(n)*self.sph_harm(m,n)

    def evaluate_derive(self, m, n, normal):
        d_r, d_theta, d_phi = self.j_inv.dot(normal)
        return self.d_hankel(n, d_r)*self.sph_harm(m,n) + \
            self.hankel(n)*self.d_sph_harm(m,n,d_theta,d_phi)
        
    def neumann_reset(self, normal):
        self.neumann = []
        for n in range(self.scale):
            for m in range(-n,n+1):
                self.neumann.append(self.evaluate_derive(m,n,normal))
        self.neumann = np.asarray(self.neumann)

    @staticmethod
    @jit(nopython=True)
    def dirichlet_reset_(scale, h, coeff, p, phi):
        dirichlet = []
        for n in range(scale):
            for m in range(-n,n+1):
                Y = coeff[abs(m),n]**0.5*p[abs(m),n]*np.exp(1j*abs(m)*phi)
                if m < 0:
                    Y = (-1)**m*Y.conjugate()
                dirichlet.append(h[n]*Y)
        return np.asarray(dirichlet)

    def dirichlet_reset(self):
        self.dirichlet = self.dirichlet_reset_(self.scale, self.h, self.coeff, self.p, self.phi)
        

