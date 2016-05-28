import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import numpy.matlib


def train_data(N=300):
    X=np.array(np.random.uniform(low=0,high=4*np.math.pi,size=[N,1]))
    f_sin=np.vectorize(sp.sin,otypes=[np.float])
    f_cos=np.vectorize(sp.cos,otypes=[np.float])
    Y=30*f_sin(X)+30*f_cos(2*X+4)+9*np.array(sp.randn(N,1))
    return [X,Y]


def SE_K(X1,X2, sigma=1,len_sc=1):
    X1_mat=np.matlib.repmat(X1,1,len(X2))
    X2_mat=np.matlib.repmat(X2,1,len(X1))
    print(X1_mat.shape); print(X2_mat.shape)
    K0 = np.square(X1_mat) + np.square(X2_mat.T) - 2 * X1 * X2.T
    K = np.power(np.exp(-1.0 / len_sc**0.2), K0)
    return np.matrix(K)

[X,Y]=train_data()
plt.scatter(X,Y)
#plt.show()


def post_pred(X,X_m,y_tr):
    K_xm=SE_K(X,X_m)
    K_mm_I=(SE_K(X_m,X_m)+1*np.eye(len(X_m))).I
    K_xx=SE_K(X,X)-K_xm*K_mm_I*K_xm.T
    mu_x=K_xm*K_mm_I*y_tr
    return[mu_x, K_xx]

print(SE_K(X,X))

ans=[]
z= np.array([np.linspace(0,4*np.math.pi,num=400)]).T
[mu,cov]=post_pred(z,X,Y)
plt.plot(z,mu)
plt.show()

'''
s=tf.Session()
z= np.array([np.linspace(0,4*np.math.pi,num=400)]).T
[mu,cov]=post_pred(z,X,Y,s)
print(cov)
errors= np.array([[2*(np.math.sqrt(cov[i,i])+sig) for i in range(len(cov))] ]).T
print (z.shape);print(errors.shape);print(mu.shape)
plt.plot()
plt.plot(z,mu)
plt.scatter(X,Y)
plt.plot(z,mu-errors)
plt.plot(z,mu+errors)
plt.show()
s.close()
'''