import numpy as np
import scipy as sp

#X,Y points from trining data
def train_data(N=300):
    sig=3
    X=np.reshape(np.array(np.linspace(0,4000*np.math.pi,num=N)),[N,1])
    f_sin=np.vectorize(sp.sin,otypes=[np.float])
    f_cos=np.vectorize(sp.cos,otypes=[np.float])
    Y=30*f_sin(X)+30*f_cos(2*X+4)+sig*np.array(sp.randn(N,1))
    Y=Y*10
    return [X,Y]


def step_data2(N):
    X_train = np.reshape(np.linspace(0, 10, num=N),(N,1))
    y = X_train.copy()
    y[y < 5.0] = 0.0
    y[y > 5.0] = 20.0
    y=(y + 0.5*np.random.randn(X_train.shape[0], 1))*10
    X_odd=X_train[1:N:2]
    X_even=X_train[0:N:2]
    Y_odd=y[1:N:2]
    Y_even=y[0:N:2]
    return [X_train,y,X_odd,Y_odd,X_even,Y_even]

def step_data(N):
    X_train = np.reshape(np.linspace(0, 100, num=N),(N,1))
    y = X_train.copy()
    y[y < 50.0] = 0.0
    y[y > 50.0] = 20.0
    y=(y + 0.5*np.random.randn(X_train.shape[0], 1))*10
    return [X_train,y]


