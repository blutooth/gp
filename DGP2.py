import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import helper as h
import Data
import maxKernStats as KS
import BayesianGPLVM as GPLVM
# Data
M=5;N=300
[X,Y,X_odd,Y_odd,X_even,Y_even]=Data.step_data2(N)
Y_odd=tf.constant(Y_odd,dtype=tf.float32)
X_even=tf.constant(X_even,dtype=tf.float32)
Y_even=tf.constant(Y_even,dtype=tf.float32)

s=tf.Session()
GPLVM.set(s)
h.set_sess(s)
Xtr=tf.constant(X,shape=[N,1],dtype=tf.float32)
Ytr=tf.constant(Y,shape=[N,1],dtype=tf.float32)

def predict(K_mn,sigma,K_mm,K_nn,Ytr):
    # predicitions
    N=h.get_dim(K_nn,0)
    K_nm=tf.transpose(K_mn)
    Sig_Inv=1e-1*np.eye(M)+K_mm+K_mnnm_2/tf.square(sigma)
    mu_post=h.Mul(tf.matrix_solve(Sig_Inv,K_mn),Ytr)/tf.square(sigma)
    mean=h.Mul(K_nm,mu_post)
    variance=K_nn-h.Mul(K_nm,h.safe_chol(K_mm,K_mn))+h.Mul(K_nm,tf.matrix_solve(Sig_Inv,K_mn))
    var_terms=2*tf.sqrt(tf.reshape(tf.diag_part(variance)+tf.square(sigma),[N,1]))
    return mean, var_terms



# layer 1
X_m_1=tf.Variable(tf.random_uniform([M,1],minval=0,maxval=15),name='X_m',dtype=tf.float32)
sigma_1=tf.Variable(tf.ones([1,1]),dtype=tf.float32,name='sigma')
noise_1=tf.Variable(tf.ones([1,1]),dtype=tf.float32,name='sigma')
len_sc_1=tf.Variable(tf.ones([1,1]),dtype=tf.float32)
K_nm_1=h.tf_SE_K(Xtr,X_m_1,len_sc_1,noise_1)
K_mm_1=h.tf_SE_K(X_m_1,X_m_1,len_sc_1,noise_1)
K_nn_1=h.tf_SE_K(Xtr,Xtr,len_sc_1,noise_1)
K_mn_1=h.tf.transpose(K_nm_1)
Tr_Knn_1=tf.trace(K_nn_1)
#mean1,var1=predict(K_mn_1,sigma_1,K_mm_1,K_nn_1)


# layer 2
h_mu=tf.Variable(np.ones((N,1)),dtype=tf.float32)
h_mu_even=tf.slice(h_mu,[0,0],[N/2,1])
h_mu_odd=tf.slice(h_mu,[N/2,0],[N/2,1])

h_S_std=tf.Variable(tf.ones([N,1]),dtype=tf.float32)
h_S=tf.square(h_S_std)
h_S_even=tf.slice(h_S,[0,0],[N/2,1])
h_S_odd=tf.slice(h_S,[N/2,0],[N/2,1])

X_m_2=tf.Variable(tf.random_uniform([M,1],minval=-5,maxval=5),dtype=tf.float32)
sigma_2=tf.Variable(tf.ones([1,1]),dtype=tf.float32)
noise_2=tf.Variable(tf.ones([1,1]),dtype=tf.float32)
len_sc_2=tf.Variable(tf.ones([1,1]),dtype=tf.float32)
Tr_Knn, K_nm_2, K_mnnm_2 = KS.build_psi_stats_rbf(X_m_2,tf.square(noise_2), len_sc_2, h_mu_odd, h_S_odd)
Tr_Knn_even, K_nm_2_even, K_mnnm_2_even = KS.build_psi_stats_rbf(X_m_2,tf.square(noise_2), len_sc_2, h_mu_even, h_S_even)


K_mm_2=h.tf_SE_K(X_m_2,X_m_2,len_sc_2,noise_2)
K_nn_2=h.tf_SE_K(h_mu_odd,h_mu_odd,len_sc_2,noise_2)
K_mn_2=h.tf.transpose(K_nm_2)
K_mn_2_even=h.tf.transpose(K_nm_2_even)
K_nn_2_even=h.tf_SE_K(h_mu_even,h_mu_even,len_sc_2,noise_2)



F_v_1=GPLVM.Bound1(h_mu,h_S,K_mm_1,K_nm_1,Tr_Knn_1,sigma_1)
s.run(tf.initialize_all_variables())
F_v_2=GPLVM.Bound2(Tr_Knn,K_nm_2,K_mnnm_2,sigma_2,K_mm_2,Y_odd)


mean,var_terms=predict(K_mn_2_even,sigma_2,K_mm_2,K_nn_2_even,Y_even)
mean=tf.reshape(mean,[-1])
var_terms=tf.reshape(var_terms,[-1])

opt = tf.train.RMSPropOptimizer(0.1)
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
        print('sig,noise,len',s.run(sigma_2),s.run(noise_2),s.run(len_sc_2))
        plt.clf()

        #plt.scatter(X_m_1,np.zeros((1,M)),color='purple')

        #plt.scatter(s.run(X_m),np.zeros(M),color='red')
        plt.plot(s.run(tf.reshape(X_even,[-1])),s.run(mean),color='blue')
        plt.scatter(X,Y,color='red')
        #plt.plot(tf.reshape(X_even,[N/2]),s.run(tf.add(var_terms,mean)),color='black')
        #plt.plot(tf.reshape(X_even,[N/2]),s.run(tf.sub(mean,var_terms)),color='black')
        print('F2',s.run(F_v_2))
        #plt.scatter(X,Y,color='green')

        h.set_sess(s);GPLVM.printer()
        #plt.scatter(s.run(tf.reshape(X_m_1,[-1])),np.zeros(M),color='yellow')
        #plt.scatter(s.run(tf.reshape(X_m_2/10,[-1])),np.zeros(M),color='purple')
        #plt.scatter(X_odd,Y_odd,color='black')

        plt.pause(2)
        #print(s.run(tf.reduce_min(h.abs_diff(Z,Z)+np.eye(M,M))))
s.close()
