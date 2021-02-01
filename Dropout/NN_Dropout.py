import tensorflow.compat.v1 as tf
import numpy as np
import time
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

tf.disable_eager_execution()
tf.disable_v2_behavior()

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, 784)
x_test = x_test.reshape(-1,784)
y_train = np.eye(np.max(y_train)+1)[y_train]
y_test = np.eye(np.max(y_test)+1)[y_test]


X = tf.placeholder(tf.float32, shape= [None, 784])
Y = tf.placeholder(tf.float32, shape = [None,10])
keep_prob = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([784, 256], stddev = 1))
B1 = tf.Variable(tf.random_normal([256],  stddev = 1))
L1 = tf.add(tf.matmul(X,W1), B1)
L1 = tf.nn.dropout(L1, keep_prob)

W2 = tf.Variable(tf.random_normal([256,256], stddev = 1))
B2 = tf.Variable(tf.random_normal([256], stddev =  1))
L2 = tf.add(tf.matmul(L1, W2), B2)
L2 = tf.nn.dropout(L2, keep_prob)

W3 = tf.Variable(tf.random_normal([256,10], stddev = 1))
#B3 = tf.Variable(tf.random_normal([10]),-1,1)
L3 = tf.matmul(L2, W3)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits = L3, labels = Y))
optimizer = tf.train.AdamOptimizer(0.001).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for epoch in range(100):
    _, cost_val = sess.run([optimizer, cost], feed_dict={X:x_train, Y:y_train, keep_prob: 0.8})
    print('Epoch:', epoch+1, "Cost:", cost_val/len(x_train))

is_correct = tf.equal(tf.argmax(L3, 1), tf.argmax(Y,1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print("accuracy:", sess.run(accuracy, feed_dict = {X:x_test, Y:y_test, keep_prob:1}))
