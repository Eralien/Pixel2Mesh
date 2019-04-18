import numpy as np
import tensorflow as tf


def position_gen(n, k=3):
    if k!=3: 
        return "Shit this function cannot do this man"
    else:
        return 3*(n-2), 2*(n-2) # The first is the edges, second is faces



if __name__ == "__main__":
    indices = np.array([[0,0],[1,1]],dtype=int)
    value = np.ones([2,])
    shape = np.array([2,2])
    sparse_a = [indices, value, shape]
    sparse_a_tf = tf.sparse_placeholder(tf.float32)
    feed_dict = dict()
    feed_dict.update({sparse_a_tf:sparse_a})
    pass
    # sparse_a_tf = tf.convert_to_tensor(sparse_a)


    b_tf = tf.random_uniform([2,3], maxval=3)
    c_tf = tf.sparse_tensor_dense_matmul(sparse_a_tf, b_tf)

    init_op = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init_op)
        b_val, c_val = sess.run([b_tf, c_tf], feed_dict=feed_dict)
        print(b_val)
        print(c_val)
    pass
