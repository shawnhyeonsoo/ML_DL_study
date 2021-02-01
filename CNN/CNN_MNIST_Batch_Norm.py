import tensorflow.compat.v1 as tf
import numpy as np
import time
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

tf.disable_eager_execution()
tf.disable_v2_behavior()
#from tf.examples.tutorials.mnist import input_data
#mnist = input_data.read_data_sets('MNIST_data', one_hot = True, reshape = False)



batch_size = 32
learning_rate = 0.001

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0 , x_test/ 255.0
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test,-1)
y_train = np.eye(np.max(y_train)+1)[y_train]
y_test = np.eye(np.max(y_test)+1)[y_test]

batch_prob = tf.placeholder(tf.bool)

X = tf.placeholder(tf.float32, shape = [None,28,28,1])
Y_Label = tf.placeholder(tf.float32, shape = [None, 10])


Kernel1 = tf.Variable(tf.truncated_normal(shape = [4,4,1,4], stddev = 0.1))
Bias1 = tf.Variable(tf.truncated_normal(shape = [4],stddev = 0.1))
Conv1 = tf.nn.conv2d(X, Kernel1, strides = [1,1,1,1], padding = 'SAME') + Bias1
Conv1 = tf.layers.batch_normalization(Conv1, center = True, scale = True, training = batch_prob)
Activation1 = tf.nn.relu(Conv1)
Pool1 = tf.nn.max_pool(Activation1, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


Kernel2 = tf.Variable(tf.truncated_normal(shape = [4,4,4,8], stddev=0.1))
Bias2 = tf.Variable(tf.truncated_normal(shape = [8], stddev = 0.1))
Conv2 = tf.nn.conv2d(Pool1, Kernel2, strides = [1,1,1,1], padding = 'SAME') + Bias2
Conv2 = tf.layers.batch_normalization(Conv2, center= True, scale = True, training = batch_prob)
Activation2 = tf.nn.relu(Conv2)
Pool2 = tf.nn.max_pool(Activation2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')


W1 = tf.Variable(tf.truncated_normal(shape = [8*7*7, 10]))
B1 = tf.Variable(tf.truncated_normal(shape = [10]))
Pool2_flat = tf.reshape(Pool2, [-1,8*7*7])
OutputLayer = tf.matmul(Pool2_flat, W1) + B1

Loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y_Label, logits = OutputLayer))
train_step = tf.train.AdamOptimizer(0.01).minimize(Loss)
correct_prediction = tf.equal(tf.argmax(OutputLayer,1), tf.argmax(Y_Label,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

with tf.Session() as sess:
    print('Start...')
    start = time.time()
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        total_batch = int(len(x_train)/batch_size)
        for j in range(total_batch):
            trainingData, Y = x_train[j:j+2], y_train[j:j+2]
            sess.run(train_step, feed_dict = {X:trainingData, Y_Label : Y, batch_prob: True})
        print("Accuracy:", sess.run(accuracy, feed_dict={X: x_test, Y_Label: y_test, batch_prob: False}))
    print("time:", time.time() - start)
