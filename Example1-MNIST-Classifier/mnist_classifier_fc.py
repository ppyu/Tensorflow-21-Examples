import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 加载MNIST数据
mnist = input_data.read_data_sets('./MNIST_data', one_hot=True)

# 输入
x = tf.placeholder(tf.float32, [None, 784])

# 实际输出
y_ = tf.placeholder(tf.float32, [None, 10])

# 第一层全连接层
layer1_w = tf.Variable(tf.zeros([784, 128]))
layer1_b = tf.Variable(tf.zeros([128]))
layer1_output = tf.nn.sigmoid(tf.matmul(x, layer1_w) + layer1_b)
# 第二层全连接层
layer2_w = tf.Variable(tf.zeros([128, 10]))
layer2_b = tf.Variable(tf.zeros([10]))
layer2_output = tf.nn.softmax(tf.matmul(layer1_output, layer2_w) + layer2_b, axis=-1)

# 定义交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(layer2_output)))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

sess = tf.InteractiveSession()
tf.global_variables_initializer().run()

# 准确率
correct_prediction = tf.equal(tf.argmax(y_, 1), tf.argmax(layer2_output, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_x, y_: batch_y})
    test_loss = cross_entropy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}, session=sess)
    test_acc = accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})
    print("train_step {},\ttest_acc:{},\ttest_loss:{} ".format(i + 1, test_acc, test_loss))

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
