import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import helper as h
import Data


# Run the Model
M=10;N=300
[X,Y]=Data.train_data(N)
s=tf.Session()
h.set_sess(s)
Xtr=tf.constant(X,shape=[N,1],dtype=tf.float32)
Ytr=tf.constant(Y,shape=[N,1],dtype=tf.float32)
X_m=tf.Variable(tf.random_uniform([M,1],minval=0,maxval=15),name='X_m')
sigma=tf.Variable(tf.ones([1,1]),dtype=tf.float32,name='sigma')
noise=tf.Variable(tf.ones([1,1]),dtype=tf.float32,name='sigma')
len_sc=tf.Variable(tf.ones([1,1]),dtype=tf.float32,name='len_sc')





s.run(tf.initialize_all_variables())
print(s.run(X_m))


X_j_m=h.jitter(X_m)

K_nm=h.tf_SE_K(Xtr,X_m,len_sc,noise)
K_mm=h.tf_SE_K(X_m,X_m,len_sc,noise)+h.tol*np.eye(M,M)
K_nn=h.tf_SE_K(Xtr,Xtr,len_sc,noise)
K_mn=h.tf.transpose(K_nm)

Sig_Inv=K_mm+h.Mul(K_mn,K_nm)/(tf.square(sigma)+h.tol)
mu_post=h.Mul(h.safe_chol(Sig_Inv,K_mn),Ytr)/(tf.square(sigma)+h.tol)
mean=h.Mul(K_nm,mu_post)
variance=K_nn-h.Mul(K_nm,h.safe_chol(K_mm,K_mn))+h.Mul(K_nm,h.safe_chol(Sig_Inv,K_mn))
var_terms=2*tf.sqrt(tf.reshape(tf.diag_part(variance)+tf.square(sigma),[N,1]))

F_v=h.F_bound(Ytr,K_mm,K_nm,K_nn,sigma)
tf.scalar_summary("F_v", F_v)

opt = tf.train.AdamOptimizer(0.1)
train=opt.minimize(-1*F_v)
init=tf.initialize_all_variables()
s.run(init)


plt.ion()
plt.axis([0, 10, -5, 5])

print('begin optimisation')
for step in xrange(10000):
    s.run(train)
    #if step % 1 == 0:
        #Z=h.jitter(X_m)
        #print(s.run(tf.reduce_min(h.abs_diff(Z,Z)+np.eye(M,M))))
        #X_m=Z
    if step % 10 == 0:
        plt.clf()
        plt.scatter(s.run(X_m),np.zeros(M),color='red')
        plt.scatter(X,s.run(mean),color='blue')
        plt.scatter(X,s.run(tf.add(var_terms,mean)),color='black')
        plt.scatter(X,s.run(tf.sub(mean,var_terms)),color='black')
        print(F_v)
        plt.scatter(X,Y,color='green')
        plt.pause(2)
        h.set_sess(s)
        #print(s.run(tf.reduce_min(h.abs_diff(Z,Z)+np.eye(M,M))))
s.close()
