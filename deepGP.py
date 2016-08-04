import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import helper as h
import Data
import maxKernStats as KS

# Data
M=20;N=300
[X,Y]=Data.train_data(N)
s=tf.Session()
h.set_sess(s)
Xtr=tf.constant(X,shape=[N,1],dtype=tf.float32)
Ytr=tf.constant(Y,shape=[N,1],dtype=tf.float32)

def predict(K_mn,sigma,K_mm,K_nn):
    # predicitions
    K_nm=tf.transpose(K_mn)
    K_mm_I=tf.matrix_inverse(K_mm)
    Sig_Inv=1e-1*np.eye(M)+K_mm+K_mnnm_2/tf.square(sigma)
    mu_post=h.Mul(tf.matrix_solve(Sig_Inv,K_mn),Ytr)/tf.square(sigma)
    mean=h.Mul(K_nm,mu_post)
    variance=K_nn-h.Mul(K_nm,h.safe_chol(K_mm,K_mn))+h.Mul(K_nm,tf.matrix_solve(Sig_Inv,K_mn))
    var_terms=2*tf.sqrt(tf.reshape(tf.diag_part(variance)+tf.square(sigma),[N,1]))
    return mean, var_terms

def predict2():
    # predicitions
    cov=h.Mul(K_mm_2,tf.matrix_inverse(K_mm_2+K_mnnm_2/tf.square(sigma_2)),K_mm_2)
    cov_chol=tf.cholesky(cov)
    mu=h.Mul(K_mm_2,tf.cholesky_solve(cov_chol,K_mn_2),Ytr)/tf.square(sigma_2)
    mean=h.Mul(K_nm_2,tf.matrix_solve(K_mm_1,mu))
    variance=K_nn_2-h.Mul(K_nm_2,h.safe_chol(K_mm_2,tf.transpose(K_nm_2)))
    var_terms=2*tf.sqrt(tf.reshape(tf.diag_part(variance)+tf.square(sigma_2),[N,1]))
    return mean, var_terms

# layer 1
X_m_1=tf.Variable(tf.random_uniform([M,1],minval=0,maxval=15),name='X_m',dtype=tf.float32)
sigma_1=tf.Variable(tf.ones([1,1]),dtype=tf.float32,name='sigma')
noise_1=tf.Variable(tf.ones([1,1]),dtype=tf.float32,name='sigma')
len_sc_1=tf.square(tf.Variable(tf.ones([1,1]),dtype=tf.float32))+0.3
K_nm_1=h.tf_SE_K(Xtr,X_m_1,len_sc_1,noise_1)
K_mm_1=h.tf_SE_K(X_m_1,X_m_1,len_sc_1,noise_1)+h.tol*np.eye(M,M)
K_nn_1=h.tf_SE_K(Xtr,Xtr,len_sc_1,noise_1)
K_mn_1=h.tf.transpose(K_nm_1)
K_mnnm_1=h.Mul(K_mn_1,K_nm_1)
Tr_Knn_1=tf.trace(K_nn_1)
#mean1,var1=predict(K_mn_1,sigma_1,K_mm_1,K_nn_1)


# layer 2
h_mu=tf.Variable(Ytr)
h_S_std=tf.Variable(np.ones((N,1)),dtype=tf.float32)
h_S=tf.square(h_S_std)
X_m_2=tf.Variable(tf.random_uniform([M,1],minval=-20,maxval=20),dtype=tf.float32)
sigma_2=tf.Variable(tf.ones([1,1]),dtype=tf.float32)
noise_2=tf.Variable(tf.ones([1,1]),dtype=tf.float32)
len_sc_2=tf.square(tf.Variable(tf.ones([1,1]),dtype=tf.float32))+0.3
'''
K_nm_2=h.tf_SE_K(h_mu,X_m_2,len_sc_2,noise_2)
K_mm_2=h.tf_SE_K(X_m_2,X_m_2,len_sc_2,noise_2)+h.tol*np.eye(M,M)
K_nn_2=h.tf_SE_K(h_mu,h_mu,len_sc_2,noise_2)
K_mn_2=h.tf.transpose(K_nm_2)
'''
Tr_Knn, K_nm_2, K_mnnm_2 = KS.build_psi_stats_rbf(X_m_2,tf.square(noise_2), len_sc_2, h_mu, h_S)

K_mm_2=h.tf_SE_K(X_m_2,X_m_2,len_sc_2,noise_2)+h.tol*np.eye(M,M)
K_nn_2=h.tf_SE_K(h_mu,h_mu,len_sc_2,noise_2)+h.tol*np.eye(N,N)
K_mn_2=h.tf.transpose(K_nm_2)

#y,S,Kmm,Knm,Kmnnm,Tr_Knn,sigma

F_v_1=h.F_bound1_v2(h_mu,h_S,K_mm_1,K_nm_1,K_mnnm_1,Tr_Knn_1,sigma_1)
#F_v_2=h.F_bound(Ytr,K_mm_2,K_nm_2,K_nn_2,sigma_2)
F_v_2=h.F_bound2_v2(Ytr,tf.zeros([N,1],dtype=tf.float32),K_mm_2,K_nm_2,K_mnnm_2,Tr_Knn,sigma_2)
mean,var_terms=predict(K_mn_2,sigma_2,K_mm_2,K_nn_2)
#mean,var_terms=predict2()
mean=tf.reshape(mean,[-1])
var_terms=tf.reshape(var_terms,[-1])

opt = tf.train.AdamOptimizer(0.01)
train=opt.minimize(-1*(F_v_1+F_v_2))
init=tf.initialize_all_variables()
s.run(init)


plt.ion()
plt.axis([0, 10, -5, 5])

print('begin optimisation')
plt.scatter(X,Y,color='red')
for step in xrange(100000):
    s.run(train)
    #if step % 1 == 0:
        #Z=h.jitter(X_m)
        #print(s.run(tf.reduce_min(h.abs_diff(Z,Z)+np.eye(M,M))))
        #X_m=Z
    if step % 50 == 0:

        print(step)
        print('sig,noise,len',s.run(sigma_1),s.run(noise_1),s.run(len_sc_1))
        plt.clf()
        #plt.scatter(s.run(X_m),np.zeros(M),color='red')
        plt.scatter(X,s.run(mean),color='blue')
        plt.scatter(X,Y,color='red')
        plt.scatter(X,s.run(tf.add(var_terms,mean)),color='black')
        plt.scatter(X,s.run(tf.sub(mean,var_terms)),color='black')
        print('F2',s.run(F_v_2))
        #plt.scatter(X,Y,color='green')
        plt.pause(2)
        h.set_sess(s)
        #print(s.run(tf.reduce_min(h.abs_diff(Z,Z)+np.eye(M,M))))
s.close()
