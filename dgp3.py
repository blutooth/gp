import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf
import helper as h
import Data
import maxKernStats as KS
import BayesianGPLVM as GPLVM
# Data
M=10;N=200
[X,Y]=Data.step_data(N)

s=tf.Session()
GPLVM.set(s)
h.set_sess(s)
Xtr=tf.constant(X,shape=[N,1],dtype=tf.float32)
Ytr=tf.constant(Y,shape=[N,1],dtype=tf.float32)

def predict(K_mn,sigma,K_mm,K_nn):
    # predicitions
    K_nm=tf.transpose(K_mn)
    Sig_Inv=1e-1*np.eye(M)+K_mm+K_mnnm_2/tf.square(sigma)
    mu_post=h.Mul(tf.matrix_solve(Sig_Inv,K_mn),Ytr)/tf.square(sigma)
    mean=h.Mul(K_nm,mu_post)
    variance=K_nn-h.Mul(K_nm,h.safe_chol(K_mm,K_mn))+h.Mul(K_nm,tf.matrix_solve(Sig_Inv,K_mn))
    var_terms=2*tf.sqrt(tf.reshape(tf.diag_part(variance)+tf.square(sigma),[N,1]))

    return mean, var_terms


def predict_new1(self):
        #self.TrKnn, self.Knm, self.Kmnnm = self.psi()
        Kmm=self.Kern.K(self.pseudo,self.pseudo)
        return  tf.matmul(self.Knm,tf.matrix_solve(Kmm+h.tol*np.eye(self.m),self.psMu))

def pred(X,X_m_1,mu,len_sc_1,noise_1):
        Kmm=h.tf_SE_K(X_m_1,X_m_1,len_sc_1,noise_1)
        Knm=h.tf_SE_K(X,X_m_1,len_sc_1,noise_1)
        posterior_mean= h.Mul(Knm,tf.matrix_solve(Kmm,mu))
        K_nn=h.tf_SE_K(X,X,len_sc_1,noise_1)
        full_cov=K_nn-h.Mul(Knm,tf.matrix_solve(Kmm,tf.transpose(Knm)))
        posterior_cov=tf.diag_part(full_cov)
        return posterior_mean,tf.reshape(posterior_cov,[N,1]),full_cov

def sample(mu, Sigma):
        N=h.get_dim(mu,0)
        rand=h.Mul(tf.cholesky(Sigma+1e-1*np.eye(N)),tf.random_normal([N,1]))+mu
        return rand

def mean_cov(K_mm,sigma,K_mn,K_mnnm,Y):
    beta=1/tf.square(sigma)
    A_I=beta*K_mnnm+K_mm
    Sig=h.Mul(K_mm,tf.matrix_solve(A_I,K_mm))
    mu=beta*h.Mul(K_mm,tf.matrix_solve(A_I,K_mn),Y)
    return  mu,Sig

# layer 1
X_m_1=tf.Variable(tf.random_uniform([M,1],minval=0,maxval=100),name='X_m',dtype=tf.float32)
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
#h_mu_odd=[tf.slice(h_mu,i) for i in ]
h_S_std=tf.Variable(np.ones((N,1)),dtype=tf.float32)
h_S=tf.square(h_S_std)
X_m_2=tf.Variable(tf.random_uniform([M,1],minval=-5,maxval=5),dtype=tf.float32)
sigma_2=tf.Variable(tf.ones([1,1]),dtype=tf.float32)
noise_2=tf.Variable(tf.ones([1,1]),dtype=tf.float32)
len_sc_2=tf.Variable(tf.ones([1,1]),dtype=tf.float32)

Tr_Knn, K_nm_2, K_mnnm_2 = KS.build_psi_stats_rbf(X_m_2,tf.square(noise_2), len_sc_2, h_mu, h_S)

K_mm_2=h.tf_SE_K(X_m_2,X_m_2,len_sc_2,noise_2)
K_nn_2=h.tf_SE_K(h_mu,h_mu,len_sc_2,noise_2)
K_mn_2=h.tf.transpose(K_nm_2)

opti_mu1,opti_Sig1=mean_cov(K_mm_1,sigma_1,K_mn_1,h.Mul(K_mn_1,K_nm_1),h_mu)
opti_mu2,opti_Sig2=mean_cov(K_mm_2,sigma_2,K_mn_2,K_mnnm_2,Ytr)





X_mean1,X_cov1,X_real_cov1= pred(Xtr,X_m_1,opti_mu1,len_sc_1,noise_1)
X_mean2,X_cov2,X_real_cov2= pred(X_mean1,X_m_2,opti_mu2,len_sc_2,noise_2)


F_v_1=GPLVM.Bound1(h_mu,h_S,K_mm_1,K_nm_1,Tr_Knn_1,sigma_1)

F_v_2=GPLVM.Bound2(Tr_Knn,K_nm_2,K_mnnm_2,sigma_2,K_mm_2,Ytr)



mean,var_terms=predict(K_mn_2,sigma_2,K_mm_2,K_nn_2)
mean=tf.reshape(mean,[-1])
var_terms=tf.reshape(var_terms,[-1])

# Sampling Routine
def sample_all():
    out=sample(X_mean1,X_real_cov1)
    X_meanS,blah,X_covS= pred(out,X_m_2,opti_mu2,len_sc_2,noise_2)
    final=sample(X_meanS,X_covS)
    return final



global_step = tf.Variable(0, trainable=False)
starter_learning_rate = 0.1
lrate=tf.train.exponential_decay(starter_learning_rate, global_step,
                                           500, 0.98, staircase=True)



opt = tf.train.GradientDescentOptimizer(lrate)
#train=opt.minimize(-1*(F_v_1+F_v_2))
learning_step = (
    tf.train.RMSPropOptimizer(lrate)
    .minimize(-1*(F_v_1+F_v_2), global_step=global_step)
)

init=tf.initialize_all_variables()
s.run(init)


plt.ion()
plt.axis([0, 10, -5, 5])

print('begin optimisation')
plt.scatter(X,Y,color='red')
for step in xrange(100000):
    s.run(learning_step)
    #if step % 1 == 0:
        #Z=h.jitter(X_m)
        #print(s.run(tf.reduce_min(h.abs_diff(Z,Z)+np.eye(M,M))))
        #X_m=Z
    if step % 500 == 0:

        print(step)
        print(s.run(lrate))
        print('sig,noise,len',s.run(sigma_2),s.run(noise_2),s.run(len_sc_2))
        plt.clf()
        plt.scatter(X,s.run(mean),color='blue')
        #plt.scatter(s.run(tf.reshape(X_m_1,[M])),s.run(tf.reshape(opti_mu2,[M])),color='orange')
        #plt.scatter(X_m_1,np.zeros((1,M)),color='purple')
        plt.scatter(s.run(Xtr),s.run(X_mean2))
        #plt.scatter(s.run(X_m),np.zeros(M),color='red')
        plt.plot(X,s.run(mean),color='blue')
        plt.scatter(X,Y,color='red')
        #plt.plot(X,s.run(tf.add(var_terms,mean)),color='black')
        #plt.plot(X,s.run(tf.sub(mean,var_terms)),color='black')

       # plt.plot(s.run(Xtr),s.run(X_mean2+2*tf.sqrt(X_cov2+X_cov1)),color='green')
        #plt.plot(s.run(Xtr),s.run(X_mean2-2*tf.sqrt(X_cov2+X_cov1)),color='green')
        print('F2',s.run(F_v_2))
        #plt.scatter(X,Y,color='green')

        h.set_sess(s);GPLVM.printer()
        plt.scatter(s.run(tf.reshape(X_m_2/10,[-1])),np.zeros(M),color='purple')
        #plt.scatter(X_odd,Y_odd,color='black')
        #plt.scatter(s.run(tf.reshape(X_m_1,[-1])),np.zeros(M),color='black')
        plt.plot(s.run(Xtr),s.run(sample_all()),color='black')

        plt.pause(2)
        #print(s.run(tf.reduce_min(h.abs_diff(Z,Z)+np.eye(M,M))))
s.close()
