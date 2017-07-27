import tensorflow as tf
import numpy as np
import time





gpu_options = tf.GPUOptions(allow_growth=True)
model = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
i = tf.constant(0)
j = tf.constant(100)
while_condition = lambda i: tf.less(i, j)
def body(i):
    c = i
    cc = sess.run(c.eval())
    print(cc)
    return [tf.add(i, 1)]

# do the loop:
r = tf.while_loop(while_condition, body, [i])
