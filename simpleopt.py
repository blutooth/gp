import tensorflow as tf
import numpy as np

a=tf.Variable(tf.convert_to_tensor(np.array([[np.float32(1.0),np.float32(2.0)]])),name='a')
y=tf.transpose(a)*np.array([[7.0,-3.0],[5.0,5.0]])*a

def get_y(a):
    a=s.run(a)
    y=a.T*np.array([[7.0,-3.0],[5.0,5.0]])*a
    return y


s=tf.Session()
s.run(tf.initialize_all_variables())

opt = tf.train.GradientDescentOptimizer(learning_rate=0.1)
train=opt.minimize(get_y(a),var_list=[a])


for step in xrange(20001):
    s.run(train)
    if step % 200 == 0:
        print(step, s.run(a))

print(s.run(a))