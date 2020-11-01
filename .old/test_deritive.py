from analysis import special
import numpy as np
import random
poles = special.Multipole(10)

x = np.random.rand(3)
n = np.random.rand(3)
k = random.random()*10
poles.reset(k,x)


m2 = random.randint(0,9)
m1 = random.randint(0,m2)

print(poles.evaluate_derive(m1,m2,n))

y1 = poles.evaluate(m1,m2)

eps = 1e-10
poles.reset(k, x+n*eps)
y2 = poles.evaluate(m1,m2)
print((y2-y1)/eps)