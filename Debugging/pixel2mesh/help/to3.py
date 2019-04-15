import tensorflow as tf
import numpy as np

# input1 = tf.constant([[2, 1, 0], [0, 2, 1]], dtype=np.int32)
# input2 = tf.Variable(tf.random_uniform([2, 3]))
# input3 = tf.Variable(np.array(range(24)).reshape(6, 4))
# output = tf.add_n([input1, input2, input3])
# output_2 = tf.gather(input2, input1, axis=1)
# output_3 = tf.reduce_sum(input3, 1)
# output_4 = tf.tile(tf.ones([3, 2, 2]), [3, 1, 1])


inputs = tf.Variable(tf.random_uniform([156, 3]))
X = inputs[:, 0]
Y = inputs[:, 1]
Z = inputs[:, 2]

h = 248 * tf.divide(-Y, -Z) + 111.5
w = 248 * tf.divide(X, -Z) + 111.5

h = tf.minimum(tf.maximum(h, 0), 223)
w = tf.minimum(tf.maximum(w, 0), 223)

x = h/(224.0/7)
y = w/(224.0/7)

img_feat = tf.Variable(tf.random_uniform([7, 7, 512]))

x1 = tf.floor(x)
x2 = tf.ceil(x)
y1 = tf.floor(y)
y2 = tf.ceil(y)

Q11 = tf.gather_nd(img_feat, tf.stack(
    [tf.cast(x1, tf.int32), tf.cast(y1, tf.int32)], 1))

dim = 512
weights = tf.multiply(tf.subtract(x2,x), tf.subtract(y2,y))
Q11 = tf.multiply(tf.tile(tf.reshape(weights,[-1,1]),[1,dim]), Q11)
outputs = tf.add_n([Q11, Q11, Q11, Q11])

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    result  = sess.run(outputs)
    pass

