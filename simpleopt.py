import helper as h
import unittest
import tensorflow as tf
import numpy as np
from scipy.spatial.distance import pdist, squareform



def sq_diff(X,Y):
        X2 = np.multiply(X,X) # sum colums of the matrix
        Y2 = np.multiply(Y,Y)
        return X2 + Y2.T - 2 * X * Y.T

def vectorized_RBF_kernel(X, sigma,l):
    # % This is equivalent to computing the kernel on every pair of examples
    K0 = sq_diff(X,X) # sum colums of the matrix
    K = sigma**2*np.exp(-0.5*K0 / l**2)
    return K

class TestUM(tf.test.TestCase):

    def setUp(self):
        pass

    def test_h_Mul(self):
        with self.test_session():
            A1=np.array([[1,2],[3,4]]); A=tf.constant(A1,shape=[2,2],dtype=tf.float32)
            B1=np.array([[1,-2],[-5,4]]); B=tf.constant(B1,shape=[2,2],dtype=tf.float32)
            C1=np.array([[7,1],[5,4]]); C=tf.constant(C1,shape=[2,2],dtype=tf.float32)
            Z1=np.matmul(np.matmul(A1,B1),np.matmul(B1,C1));
            print(Z1)
            self.assertAllEqual(h.Mul(A,B,B,C).eval(),Z1)

    def test_get_dim(self):
        A1=np.array([[1,2],[1,5],[3,4]]); A=tf.constant(A1,dtype=tf.float32)
        self.assertEqual(h.get_dim(A,0),3)
        self.assertEqual(h.get_dim(A,1),2)

    def test_condition(self):
        with self.test_session():
            A1=np.array([[1.0,2,5],[1,5,6],[3,4,9]]); A=tf.constant(A1,dtype=tf.float32)
            A2=A+h.tol*np.eye(3)
            self.assertAllEqual(h.condition(A).eval(),A2.eval())


    def test_squared_diff(self):
        with self.test_session():
            #rows of left, columns of the other
            A=np.array([[1.0,2.0,5.0,7.0,9.0]]).T;A1=tf.constant(A,shape=[5,1],dtype=tf.float32)
            B=np.array([[3.0,4.0,6.0,7.0]]).T;B1=tf.constant(B,shape=[4,1],dtype=tf.float32)
            pairwise_dists = sq_diff(A,B)
            PD=tf.constant(pairwise_dists,shape=[5,4],dtype=tf.float32)
            PD1=h.squared_diff(A1,B1)
            self.assertAllEqual(PD.eval(), PD1.eval())


    def test_safe_chol(self):
        with self.test_session():
            B=np.array([[1.0,1.1,1.2,1.3]]).T;B1=tf.constant(B,shape=[4,1],dtype=tf.float32)
            A=h.squared_diff(B1,B1);
            A=A+0.2*np.eye(4)
            x=h.safe_chol(A,B1)
            self.assertAllClose(tf.matmul(A,x).eval(),B1.eval(),atol=1e-1)

    def test_chol_det(self):
        with self.test_session():
            B=np.array([[1.0,1.1,1.2,1.3]]).T;B1=tf.constant(B,shape=[4,1],dtype=tf.float32)
            A=h.squared_diff(B1,B1);
            A=A+0.2*np.eye(4)
            self.assertAllClose(tf.matrix_determinant(h.condition(A)).eval(),h.chol_det(A).eval(),atol=1e-3)

    def test_log_det(self):
        with self.test_session():
            B=np.array([[1.0,1.1,1.2,1.3]]).T;B1=tf.constant(B,shape=[4,1],dtype=tf.float32)
            A=h.squared_diff(B1,B1);
            A=A+0.2*np.eye(4)
            self.assertAllClose(tf.log(tf.matrix_determinant(h.condition(A))).eval(),h.log_det(A).eval(),atol=1e-3)

    def test_SE_K(self):
        with self.test_session():
            len_sc=tf.ones([1,1])
            noise=2*tf.ones([1,1])
            A=np.array([[1.0,2.0,5.0]]).T;A1=tf.constant(A,shape=[3,1],dtype=tf.float32)
            LHS=h.tf_SE_K(A1,A1,len_sc,noise)
            RHS=vectorized_RBF_kernel(A,2,1)
            self.assertAllClose(LHS.eval(),RHS,atol=1e-3)


if __name__ == '__main__':
    unittest.main()