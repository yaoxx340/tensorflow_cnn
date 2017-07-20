from skimage import io,transform
import tensorflow as tf
import numpy as np


path1 = "/home/yuan/Google Drive/Research/cv/CNN_detector/mouse/18.png"
 

object_dict = {0:'mouse',1:'bg'}

w=100
h=100
c=1

def read_one_image(path):
    img = io.imread(path,as_grey=True)
    img = transform.resize(img,(w,h))
    return np.asarray(img)

with tf.Session() as sess:
    data = []
    data1 = read_one_image(path1)
 
    data.append(data1)
    data=np.reshape(data,(1,100,100,1))
 

    saver = tf.train.import_meta_graph('/home/yuan/Google Drive/Research/cv/CNN_detector/model.ckpt.meta')
    saver.restore(sess,tf.train.latest_checkpoint('/home/yuan/Google Drive/Research/cv/CNN_detector/'))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    feed_dict = {x:data}

    logits = graph.get_tensor_by_name("logits_eval:0")

    classification_result = sess.run(logits,feed_dict)


    print(classification_result[0][1])

    print(tf.argmax(classification_result,1).eval()[0])

    output = []
    output = tf.argmax(classification_result,1).eval()
    for i in range(len(output)):
        print("no",i+1,"predixtion:"+object_dict[output[i]])
