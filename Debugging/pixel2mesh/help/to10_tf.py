import numpy as np
import tensorflow as tf

val = np.arange(99).reshape((-1,3))
idx = np.arange(2,20,2).reshape((-1,1))
idx = np.hstack((idx, np.random.randint(0, 3, size=(idx.shape))))
val_tf = tf.constant(val)
idx_tf = tf.constant(idx)
gather = tf.gather(val_tf, idx_tf)
reduced = tf.reduce_sum(gather, 1)

init_op = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init_op)
    resul_gather, result_reduced = sess.run([gather, reduced])
    # print(result)
    pass