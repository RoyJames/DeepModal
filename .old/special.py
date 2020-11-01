import numpy as np

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
        self.p_d[0][0] = 0
        for i in range(self.n_max-1):
            self.p[i+1,i+1] =  -(2*i+1)*(1-np.cos(theta)**2)**0.5*self.p[i,i]
            self.p_d[i+1,i+1] =  (2*i+1)*np.cos(theta)/(1-np.cos(theta)**2)**0.5*self.p[i,i] - (2*i+1)*(1-np.cos(theta)**2)**0.5*self.p_d[i,i]
            
        for m in range(self.n_max - 1):
            self.p[m,m+1] = np.cos(theta)*(2*m+1)*self.p[m,m]
            self.p_d[m,m+1] = (2*m+1)*self.p[m,m] + np.cos(theta)*(2*m+1)*self.p_d[m,m]
            for i in range(m+1, self.n_max - 1):
                self.p[m,i+1] = ((2*i+1)*np.cos(theta)*self.p[m,i]-(i+m)*self.p[m,i-1])/(i-m+1)
                self.p_d[m,i+1] = ((2*i+1)*self.p[m,i]+(2*i+1)*np.cos(theta)*self.p_d[m,i]-(i+m)*self.p_d[m,i-1])/(i-m+1)
        #=================== inverse Jacobian ==========================
        x,y,z = x[0],x[1],x[2]
        t1 = r*r*(x**2+y**2)**0.5
        t2 = (x**2+y**2)
        self.j_inv = np.array([
            [x/r,       y/r,        z/r     ],
            [x*z/t1],   y*z/t1,     -t2/t1  ],
            [-y/t2,     x/t2,       0       ]
        ])

    def sph_harm(self, m, n):
        Y = self.coeff[abs(m),n]**0.5*self.p[abs(m),n]*np.exp(1j*abs(m)*self.phi)
        if m < 0:
            Y = (-1)**m*Y.conjugate()
        return Y

    def d_sph_harm(self, m, n, d_theta, d_phi):
        d_Y = self.coeff[abs(m),n]**0.5*self.p_d[abs(m),n]*(-np.sin(theta)*d_theta)*np.exp(1j*abs(m)*self.phi) + \
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
        

