from scipy import special
import numpy as np
def hankel(n,z,derivative=False):
    return special.spherical_jn(n,z,derivative) + 1j*special.spherical_yn(n,z,derivative)

def my_hankel(n,z,derivative = False):
    h = np.zeros(n+2).astype(np.complex)
    h[0] = -1j/z*np.exp(1j*z)
    h[1] = -(z+1j)/z**2*np.exp(1j*z)
    for i in range(2,n+2):
        h[i] = (2*i-1)/z*h[i-1]-h[i-2]
    if derivative==False:
        return h[n]
    else:
        return -h[n+1]+n/z*h[n]

def alp(m,n,x):
    p = np.zeros((n+2,n+2)).astype(np.complex)
    p[0][0] = 1
    for i in range(m):
        p[i+1,i+1] =  -(2*i+1)*(1-x**2)**0.5*p[i,i]
    p[m,m+1] = x*(2*m+1)*p[m,m]
    for i in range(m+1, n):
        p[m,i+1] = ((2*i+1)*x*p[m,i]-(i+m)*p[m,i-1])/(i-m+1)
    return p[m,n]

def my_sph_harm(m,n,theta,phi):
    coeff = (2*n+1)/(4*np.pi)
    for i in range(2*abs(m)):
        coeff /= n-abs(m)+i+1
    coeff = coeff**0.5
    Y = coeff*alp(abs(m),n,np.cos(phi))*np.exp(1j*abs(m)*theta)
    if m < 0:
        return (-1)**m*Y.conjugate()
    else:
        return Y

def test_my_sph_harm():
    for i in range(5):
        for j in range(-i,i+1):
            theta = random.random()*np.pi
            phi = random.random()*np.pi
            print(i,j)
            print(my_sph_harm(j,i,theta,phi))
            print(special.sph_harm(j,i,theta,phi))

def test_my_hankel():
    for i in range(10):
        z = random.random()*20
        print(hankel(i,z)-my_hankel(i,z),hankel(i,z,True)-my_hankel(i,z,True))
