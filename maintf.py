import numpy as np
import scipy as sp
import matplotlib.pylab as plt
import tensorflow as tf

#X,Y points from trining data
def train_data(N=300):
    sig=3
    X=np.array(np.random.uniform(low=0,high=4*np.math.pi,size=[N,1]))
    f_sin=np.vectorize(sp.sin,otypes=[np.float])
    f_cos=np.vectorize(sp.cos,otypes=[np.float])
    Y=30*f_sin(X)+30*f_cos(2*X+4)+sig*np.array(sp.randn(N,1))
    return [X,Y]

#MAT MUL
def Mul(*arg):
    if len(arg)==2:
        return tf.matmul(arg[0],arg[1])
    else:
        return Mul(tf.matmul(arg[0],arg[1]),*arg[2:])

#Works
def get_dim(X,index):
     shape=X.get_shape()
     return int(shape[index])


def tf_SE_K(X1,X2):
    # Dimensions of X1,X2
    l1=get_dim(X1,0); l2=get_dim(X2,0)

    X2_T=tf.transpose(X2)
    X1_mat=tf.tile(X1,[1,l2])
    X2_mat=tf.tile(X2_T,[l1,1])
    K_1=tf.squared_difference(X1_mat,X2_mat)
    exponent=tf.exp(-(0.5*K_1/tf.square(1.0)))#len_sc
    K_f= tf.mul(tf.square(1.0), exponent)+tf.constant(np.eye(l1,l2),shape=[l1,l2],dtype=tf.float32) #sig
    return K_f

def log_density(x,mu,prec):
    diff=tf.sub(x,mu)
    diff_sq=-0.5*tf.matmul(tf.matmul(tf.transpose(diff),prec),diff)
    logdet=log_determinant(prec)
    return diff_sq+logdet

def Matrix_Inversion_Lemma(A,U,C_inv,V):
    A_inv=tf.matrix_inverse(A)
    z=A_inv-Mul(A_inv,U,tf.matrix_inverse(C_inv+Mul(V,A_inv,U)),V,A_inv)
    return z

def log_determinant(Z):
    logdet=tf.log(tf.matrix_determinant(Z))
    return logdet

def F_bound(y,Kmm,Knm,Knn):
    Eye=tf.constant(np.eye(N,N), shape=[N,N],dtype=tf.float32)
    sigEye=tf.mul(1.0,Eye)
    print(s.run(tf.matrix_inverse(sigEye)))
    #sigEye=tf.mul(tf.square(sigma),Eye)
    Kmn=tf.transpose(Knm)
    prec=Matrix_Inversion_Lemma(sigEye,Knm,Kmm,Kmn)
    zeros=tf.constant(np.zeros(N),shape=[N,1],dtype=tf.float32)
    log_den=log_density(y,zeros,prec)
    Kmm_inv=tf.matrix_inverse(Kmm)
    trace_term=tf.trace(Knn-Mul(Knm,Kmm_inv,Kmn))*(0.5)
    return log_den-trace_term

[X,Y]=train_data()
M=50;N=300
s=tf.Session()
Xtr=tf.constant(X,shape=[N,1],dtype=tf.float32)
Ytr=tf.constant(Y,shape=[N,1],dtype=tf.float32)
X_m=tf.Variable(tf.random_uniform([M,1],minval=0,maxval=10),name='X_m')
#sigma=tf.Variable(tf.ones([1,1]),dtype=tf.float32,name='sigma')
#len_sc=tf.Variable(tf.ones([1,1]),dtype=tf.float32,name='len_sc')
s.run(tf.initialize_all_variables())
print(s.run(X_m))


K_nm=tf_SE_K(Xtr,X_m)
K_mm=tf_SE_K(X_m,X_m)
print(s.run(K_mm))
K_nn=tf_SE_K(Xtr,Xtr)
K_mn=tf.transpose(K_nm)


SIGMA=tf.matrix_inverse(K_mm+Mul(K_mn,K_nm))
mu=Mul(K_mm,SIGMA,K_mn,Ytr)
Real_mean=Mul(K_nm,tf.matrix_inverse(K_mm),mu)


F_v=F_bound(Ytr,K_mm,K_nm,K_nn)
tf.scalar_summary("F_v", F_v)

opt = tf.train.AdagradOptimizer(1.0)
train=opt.minimize(-1*F_v)
init=tf.initialize_all_variables()
s.run(init)


plt.ion()
plt.axis([0, 10, -5, 5])

print('begin optimisation')
for step in xrange(500):
    s.run(train)
    if step % 2 == 0:
        print(step,s.run(X_m))
        plt.clf()
        plt.scatter(s.run(X_m),np.zeros(M),color='red')
        plt.scatter(X,s.run(Real_mean))
        plt.pause(2)
s.close()
