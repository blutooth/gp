import numpy as np
import scipy as sp

#X,Y points from trining data
def train_data(N=300):
    sig=3
    X=np.array(np.random.uniform(low=0,high=4*np.math.pi,size=[N,1]))
    f_sin=np.vectorize(sp.sin,otypes=[np.float])
    f_cos=np.vectorize(sp.cos,otypes=[np.float])
    Y=30*f_sin(X)+30*f_cos(2*X+4)+sig*np.array(sp.randn(N,1))
    return [X,Y]