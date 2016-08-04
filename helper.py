import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import tensorflow as tf

S=None
seed=1
tol=1e-2

#MAT MUL
def Mul(*arg):
    if len(arg)==1:
        return arg[0]
    else:
        return tf.matmul(arg[0],Mul(*arg[1:]))


def get_dim(X,index):
     shape=X.get_shape()
     return int(shape[index])

#takes cholesky factor as argument
def safe_chol(A,RHS):
    conditioned=condition((A+tf.transpose(A))/2)
    chol=tf.cholesky(conditioned)
    return tf.cholesky_solve(chol,RHS)

def isclose(X):
    Z= abs_diff(X,X)+np.eye(get_dim(X,0),get_dim(X,0))
    return S.run(tf.to_float(tf.reduce_min(Z)))< 0.01

def set_sess(Sess):
    global S
    S=Sess

def jitter(X):
       X_new=X
       while(isclose(X_new)):
        global seed
        seed=seed+1
        np.random.seed(seed)
        X_new=X_new+0.2*np.random.randn(get_dim(X,0),1)
       return X_new

def squared_diff(X1,X2):

    l1=get_dim(X1,0); l2=get_dim(X2,0)
    X2_T=tf.transpose(X2)
    X1_mat=tf.tile(X1,[1,l2])
    X2_mat=tf.tile(X2_T,[l1,1])
    K_1=tf.squared_difference(X1_mat,X2_mat)
    return K_1

def abs_diff(X1,X2):

    l1=get_dim(X1,0); l2=get_dim(X2,0)
    X2_T=tf.transpose(X2)
    X1_mat=tf.tile(X1,[1,l2])
    X2_mat=tf.tile(X2_T,[l1,1])
    K_1=tf.abs(tf.sub(X1_mat,X2_mat))
    return K_1

def chol_det(X):
    conditioned=condition(X)
    return tf.square(tf.assert_greater(tf.reduce_prod(tf.diag_part(tf.cholesky(conditioned)))))

def condition(X):
    return (X+tf.transpose(X))/2+tol*np.eye(get_dim(X,0))

def log_det(Z):
    #conditioned=condition(Z)
    Z=(Z+tf.transpose(Z))/2
    return 2*tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(Z))))

    chol=tf.cholesky(Z)
    logdet=2*tf.reduce_sum(tf.log(tf.diag_part(chol)))
    return logdet

def lambda_Eye(lamb,N):
    Eye=tf.constant(np.eye(N,N), shape=[N,N],dtype=tf.float32)
    sigEye=tf.mul(lamb,Eye)
    return sigEye

### more related to model
def tf_SE_K(X1,X2,len_sc,noise):
    X11=X1
    X22=X2
    sq_diff= squared_diff(X11,X22)
    K=tf.square(noise)*tf.exp(-(0.5*sq_diff/tf.square(len_sc)))#len_sc
    if get_dim(X1,0)==get_dim(X2,0):
        return K+tol*np.eye(get_dim(X1,0))
    else:
        return K

##unit tests so far

#logdet more relevant
def log_det_lemma(ldz,Z_I,U,W_I,V_T):
    LDZ=ldz-log_det(W_I)
    return LDZ+log_det(W_I+Mul(V_T,Z_I,U))

def log_density(x,mu,prec,logdetcov):
    diff=x-mu
    diff_sq=-0.5*tf.matmul(tf.matmul(tf.transpose(diff),prec),diff)
    return diff_sq-0.5*logdetcov

#corresponds to formula for (z+UWV^T)^-1
def Matrix_Inversion_Lemma(Z_I, U,W_I,V_T):
    A=W_I+Mul(V_T,Z_I,U)
    prod=safe_chol(W_I+Mul(V_T,Z_I,U),Mul(V_T,Z_I))
    return Z_I -Mul(Z_I,U,prod)


def F_bound(y,Kmm,Knm,Knn,sigma):
    #matrices to be used
    N=get_dim(Knn,0)
    sig_sq=tf.square(sigma)
    sigEye=lambda_Eye(sig_sq,N)
    sigEye_I=lambda_Eye(1/sig_sq,N)
    zeros=tf.constant(np.zeros(N),shape=[N,1],dtype=tf.float32)
    Kmn=tf.transpose(Knm)
    #main calcz
    prec=Matrix_Inversion_Lemma(sigEye_I,Knm,Kmm,Kmn)
    log_det_cov=log_det_lemma(tf.log(sig_sq)*N,sigEye_I,Knm,Kmm,Kmn)
    log_den=log_density(y,zeros,prec,log_det_cov)
    trace_term=tf.trace(Knn-Mul(Knm,safe_chol(Kmm,Kmn)))
    return log_den-trace_term

'''
def F_bound_v1(y,S,Kmm,Knm,Knn,sigma):
    #matrices to be used
    Kmnnm=Mul(tf.transpose(Knm),Knm)
    N=get_dim(y,0)
    Q_nn=np.eye(N)*tf.square(sigma)+Mul(Knm,safe_chol(Kmm,tf.transpose(Knm)))
    Kmm_chol=tf.cholesky(Kmm)
    bound=-0.5*(tf.trace(Knn)+tf.trace(tf.cholesky_solve(Kmm_chol,Kmnnm)))/tf.square(sigma)
    bound+=multivariate_normal(y, tf.zeros([N,1],dtype=tf.float32), tf.cholesky(Q_nn))
    bound+=tf.reduce_sum(tf.log(S))
    return bound
'''

def F_bound1_v2(y,S,Kmm,Knm,Kmnnm,Tr_Knn,sigma):
    #matrices to be used
    N=get_dim(y,0)
    Kmm_chol=tf.cholesky(Kmm)
    Q_nn=tf.square(sigma)*np.eye(N)+Mul(Knm,tf.cholesky_solve(Kmm_chol,tf.transpose(Knm)))
    bound=-0.5*(Tr_Knn-tf.trace(tf.cholesky_solve(Kmm_chol,Kmnnm)))/tf.square(sigma)
    bound+=multivariate_normal(y, tf.zeros([N,1],dtype=tf.float32), tf.cholesky(Q_nn))
    return bound-0.5*tf.reduce_sum(S)/tf.square(sigma)+0.5*tf.reduce_sum(tf.log(S))
def multivariate_normal(x, mu, L):
    """
    L is the Cholesky decomposition of the covariance.
    x and mu are either vectors (ndim=1) or matrices. In the matrix case, we
    assume independence over the *columns*: the number of rows must match the
    size of L.
    """
    d = x - mu
    alpha = tf.matrix_triangular_solve(L, d, lower=True)
    num_col = 1 if tf.rank(x) == 1 else tf.shape(x)[1]
    num_col = tf.cast(num_col, tf.float32)
    num_dims = tf.cast(tf.shape(x)[0], tf.float32)
    ret = - 0.5 * num_dims * num_col * np.log(2 * np.pi)
    ret += - num_col * tf.reduce_sum(tf.log(tf.diag_part(L)))
    ret += - 0.5 * tf.reduce_sum(tf.square(alpha))
    return tf.reduce_sum(ret)

def F_bound2(y,S,Kmm,Knm,Kmnnm,Tr_Knn,sigma):
    #matrices to be used
    Kmn=tf.transpose(Knm)
    N=get_dim(Knm,0)
    sig_sq=tf.square(sigma)
    sigEye=lambda_Eye(sig_sq,N)
    sigEye_I=lambda_Eye(1/sig_sq,N)
    zeros=tf.constant(np.zeros(N),shape=[N,1],dtype=tf.float32)
    #main calcz
    prec=Matrix_Inversion_Lemma(sigEye_I,Knm,Kmm,Kmn)
    log_det_cov=log_det_lemma(tf.log(sig_sq)*N,sigEye_I,Knm,Kmm,Kmn)
    log_den=log_density(y,zeros,prec,log_det_cov)
    trace_term=Tr_Knn-tf.trace(safe_chol(Kmm,Kmnnm))
    return log_den-0.5*(trace_term+tf.reduce_sum(S))/tf.square(sigma)


def F_bound2_v2(y,S,Kmm,Knm,Kmnnm,Tr_Knn,sigma):
    #matrices to be used
    N=get_dim(y,0)
    Kmm_chol=tf.cholesky(Kmm)
    Q_nn=tf.square(sigma)*np.eye(N)+Mul(Knm,tf.cholesky_solve(Kmm_chol,tf.transpose(Knm)))
    bound=-0.5*(Tr_Knn-tf.trace(tf.cholesky_solve(Kmm_chol,Kmnnm)))/tf.square(sigma)
    bound+=multivariate_normal(y, tf.zeros([N,1],dtype=tf.float32), tf.cholesky(Q_nn))
    return bound