import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import numpy.matlib
import tensorflow as tf
import os
from matplotlib import pyplot as pl
from scipy.stats import multivariate_normal as mvn


sig=3

def train_data(N=1000):
    X=np.array(np.random.uniform(low=0,high=4*np.math.pi,size=[N,1]))
    f_sin=np.vectorize(sp.sin,otypes=[np.float])
    f_cos=np.vectorize(sp.cos,otypes=[np.float])
    Y=30*f_sin(X)+30*f_cos(2*X+4)+sig*np.array(sp.randn(N,1))
    return [X,Y]

#end of prior funcs


def tf_SE_K(X1,X2,sigma, len_sc):
    (l1,lol)=X1.get_shape()
    (l2,lol)=X2.get_shape()
    l2=int(l2)
    l1=int(l1)
    print(l2)
    X1_mat=tf.tile(X1,[1,l2])
    X2_mat=tf.transpose(tf.tile(X2,[1,l1]))
    K_1=tf.add(tf.square(X1_mat),tf.square(X2_mat))
    K_2=tf.mul(tf.constant(2.0, shape=[l1,l2]),tf.matmul(X1,tf.transpose(X2)))
    K_3=tf.sub(K_1,K_2)
    exponent=tf.pow(tf.exp(tf.div(tf.constant(-1.0,shape=[l1,l2]),tf.tile(len_sc,[l1,l2]))),K_3)
    K_f= tf.mul(tf.square(sigma), exponent)
    return K_f

def MVN_density(x,mu,cov):
    (l1,lol)=x.get_shape()
    print(x.get_shape()); print(s.run(x));print(s.run(mu))
    diff=tf.sub(x,mu)
    diff_sq=-1.0*tf.matmul(tf.matmul(tf.transpose(diff),cov),diff)/2.0
    print(s.run(diff_sq))
    print(s.run(cov))
    print(s.run(tf.matrix_determinant(cov)))
    # to do: decompose diagonal to get cov
    #logdet=-1.0*tf.(tf.log(tf.slice(tf.self_adjoint_eig(cov),[1],)))
    return diff_sq

[X,Y]=train_data()




# Create an optimizer.

M=30
s=tf.Session()
Xtr=tf.constant(X,shape=X.shape,dtype=tf.float32)
Ytr=tf.constant(Y,shape=Y.shape,dtype=tf.float32)
X_m=tf.Variable(tf.random_uniform(minval=0,maxval=10,shape=[M,1]),name='X_m')

sigma=tf.Variable(tf.ones([1,1]),dtype=tf.float32,name='sigma')
len_sc=tf.Variable(tf.ones([1,1]),dtype=tf.float32,name='len_sc')

K_xm=tf_SE_K(Xtr,X_m,sigma,len_sc)
K_mm=tf_SE_K(X_m,X_m,sigma,len_sc)

Q_nn=tf.matmul(K_xm, tf.matmul(tf.matrix_inverse(K_mm),tf.transpose(K_xm)))
F_v=MVN_density(Ytr,tf.zeros(shape=Y.shape,dtype=tf.float32),Q_nn)
grad=tf.gradients(F_v,sigma)
opt = tf.train.AdamOptimizer(1e-4).minimize(F_v)
# Compute the gradients for a list of variables.
#grads_and_vars = opt.compute_gradients(loss, <list of variables>)

# grads_and_vars is a list of tuples (gradient, variable).  Do whatever you
# need to the 'gradient' part, for example cap them, etc.
#capped_grads_and_vars = [(MyCapper(gv[0]), gv[1]) for gv in grads_and_vars]

# Ask the optimizer to apply the capped gradients.
#opt.apply_gradients(capped_grads_and_vars)
tf.initialize_all_variables()
print('begin optimisation')
for step in xrange(201):
    s.run(opt)
    if step % 20 == 0:
        print(step, s.run(sigma),s.run(len_sc))

s.close()

