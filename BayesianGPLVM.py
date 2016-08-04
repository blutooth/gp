import tensorflow as tf
import helper as h
import numpy as np
# NB SIgmA_Y is sqrt

s=None
def set(Sess):
    global s
    s=Sess

def printer():
    print(s.run(matrix_determinant))

def Bound1(y,S,Kmm,Knm,Tr_Knn,sigma):
#matrices to be used
    Kmm_chol=tf.cholesky(Kmm)
    sig_2=tf.square(sigma)
    N=h.get_dim(y,0)
    Q_nn=h.Mul(Knm,tf.cholesky_solve(Kmm_chol,tf.transpose(Knm)))
    Q_I_chol=tf.cholesky(sig_2*np.eye(N)+Q_nn)
    bound=-0.5*(Tr_Knn-Q_nn)/sig_2
    bound+=h.multivariate_normal(y, tf.zeros([N,1],dtype=tf.float32), Q_I_chol)
    bound-=0.5*tf.reduce_sum(S)/sig_2+0.1*0.5*tf.reduce_sum(tf.log(S))
    return bound

def Bound2(phi_0,phi_1,phi_2,sigma_noise,K_mm,mean_y):
    # Preliminary Bound
    beta=1/tf.square(sigma_noise)
    bound=0
    N=h.get_dim(mean_y,0)
    M=h.get_dim(K_mm,0)
    W_inv_part=beta*phi_2+K_mm
    global phi_200
    phi_200=tf.matrix_solve(W_inv_part,tf.transpose(phi_1))
    W=beta*np.eye(N)-tf.square(beta)*h.Mul(phi_1,tf.matrix_solve(W_inv_part,tf.transpose(phi_1)))
    # Computations
    bound+=N*tf.log(beta)
    bound+=h.log_det(K_mm+1e-3*np.eye(M))
    bound-=h.Mul(tf.transpose(mean_y),W,mean_y)
    global matrix_determinant
    matrix_determinant=tf.ones(1) #h.log_det(W_inv_part+1e2*np.eye(M))#-1e-40*tf.exp(h.log_det(W_inv_part))


    bound-=h.log_det(W_inv_part+1e-3*tf.reduce_mean(W_inv_part)*np.eye(M))
    bound-=beta*phi_0
    bound+=beta*tf.trace(tf.cholesky_solve(tf.cholesky(K_mm),phi_2))
    bound=bound*0.5
    return bound