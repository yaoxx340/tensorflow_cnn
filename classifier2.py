from skimage.transform import pyramid_gaussian
from skimage.io import imread
from skimage import data
import numpy as np
from skimage import novice
from sklearn.externals import joblib
import cv2
import glob
from sklearn.utils.extmath import fast_dot
import argparse as ap
from nms import nms
from skimage import io,transform
import tensorflow as tf
import numpy as np



gpu_options = tf.GPUOptions(allow_growth=True)
model = tf.global_variables_initializer()
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
saver = tf.train.import_meta_graph('/home/yaoxx340/CNN_detector/model.ckpt.meta')
saver.restore(sess,tf.train.latest_checkpoint('/home/yaoxx340/CNN_detector/'))
graph = tf.get_default_graph()
xxx = graph.get_tensor_by_name("x:0")
logits = graph.get_tensor_by_name("logits_eval:0")

#object_dict = {0:'mouse',1:'bg'}

w = 100
h = 100
c = 1
min_wdw_sz = [91, 161]
halfx = (min_wdw_sz[0]-1)/2
halfy = (min_wdw_sz[1]-1)/2
step_size = (10, 10)
#pts = np.mgrid[890:1400:step_size[0], 250:730:step_size[1],0.0: 6.28: 5].reshape(3,-1).T
#length = tf.constant(pts.shape[0])
#length = tf.reshape(length,(1,1))


filename = "test/83.png"
strings = filename[-6:-4]
im = imread(filename)
img = imread(filename,as_grey=True)
#tf_image = tf.convert_to_tensor(img)
#tf_pts = points = tf.convert_to_tensor(pts, dtype=tf.float32)
locations = []
detections = []
# Create a TensorFlow Variable
#image =tf.convert_to_tensor(img)
#points = tf.convert_to_tensor(pts, dtype=tf.float32)

def read_one_image(im):
    img = transform.resize(im,(w,h))
    return np.asarray(img)
def sliding_window(image, window_size,step_size):    
    pair = np.mgrid[890:1400:step_size[0], 250:730:step_size[1],0.0: 6.28: 0.147].reshape(3,-1).T
    for pts in pair:
        xx = int(pts[0])
        yy = int(pts[1])
        tt = pts[2]
        #crop = image[yy-halfy:yy+halfy+1,xx-halfx:xx+halfx+1]
        crop = np.mgrid[yy-halfy:yy+halfy+1, xx-halfx:xx+halfx+1].reshape(2,-1).T
        crop[:,[0, 1]] = crop[:,[1, 0]]
        col = crop.shape[0]
        newp = np.ones((col,3))
        newp[:,:-1] = crop
        transform = np.array([[np.cos(tt),-np.sin(tt),-xx*np.cos(tt)+xx+yy*np.sin(tt)],[np.sin(tt),np.cos(tt),-xx*np.sin(tt)-yy*np.cos(tt)+yy],[0,0,1]])
        newp = fast_dot(transform,newp.T)
        newp = newp[0:2,:]
        newp = np.transpose(newp).astype(int)
        imx = newp[:,0]
        imy = newp[:,1]
        newim = np.array(image[imy, imx]).reshape((window_size[1],window_size[0]))
        #print(xx,yy,tt)
        newim = read_one_image(newim)
        #newim = np.float16(newim)
        #print(xx,yy,newim.shape)
        locations.append((xx,yy,tt))
        detections.append(newim)

sliding_window(img, min_wdw_sz, step_size)
length = len(detections)
f = lambda A, n=200: [A[i:i+n] for i in range(0, len(A), n)]
lst_detection = f(detections)
lst_locations = f(locations)
max_prob = []
max_locations = []
max_index = []
#print(len(lst))

for i in lst_detection:
    data=np.reshape(i,(len(i),100,100,1))
    feed_dict = {xxx:data}
    classification_result = sess.run(logits,feed_dict)
    #prob = classification_result[:,0]/(classification_result[:,0]+classification_result[:,1])
    prob = classification_result[:,0] 
    max_ind = np.argmax(prob)
    max_prob.append(prob[max_ind])
    max_index.append(max_ind)
#max_prob[5.6,4,8,0,-5,7]
#max_index[4,5,3,0,3,2]
print(max_prob)
mmax_ind = np.argmax(max_prob)
lst_index = max_index[mmax_ind]
max_pts = lst_locations[mmax_ind][lst_index]
print(max_pts)



x_tl = max_pts[0]
y_tl = max_pts[1] 
theta = max_pts[2]
lu = [x_tl-halfx,y_tl-halfy]
ll = [x_tl-halfx,y_tl+halfy]
rl = [x_tl+halfx,y_tl+halfy]
ru = [x_tl+halfx,y_tl-halfy]
transform = np.array([[np.cos(theta),-np.sin(theta),-x_tl*np.cos(theta)+x_tl+y_tl*np.sin(theta)],[np.sin(theta),np.cos(theta),-x_tl*np.sin(theta)-y_tl*np.cos(theta)+y_tl],[0,0,1]])
p1=np.append(lu,1)
p2=np.append(ll,1)
p3=np.append(rl,1)
p4=np.append(ru,1)
p=np.array([p1,p2,p3,p4])
p=np.transpose(p)
result=np.dot(transform,p)
result=result[0:2,:]
result=np.transpose(result)
point=result.astype(int)
lu=point[0]
ll=point[1]
rl=point[2]
ru=point[3]
pts = np.array([lu,ll,rl,ru], np.int32)
pts1 = pts.reshape((-1,1,2))
cv2.polylines(im,[pts1],True,(0,255,0))
    #cv2.imshow("Final Detections after applying NMS", clone)
cv2.imwrite("/home/yaoxx340/CNN_detector/result/"+str(strings)+"final.png",im)
    #cv2.waitKey()
'''

i = tf.constant(0)
while_condition = lambda i: tf.less(i, length)
def loop_body(i):
    tf.Print(i,[i])
    xx = tf_pts[i][0]
    yy = tf_pts[i][1]
    tt = tf_pts[i][2]
    x_array = tf.range(xx-halfx,xx+halfx+1)
    y_array = tf.range(yy-halfy,yy+halfy+1)
    X, Y = tf.meshgrid(x_array, y_array)
    X = tf.reshape(X, [-1])
    Y = tf.reshape(Y, [-1])
    one_shape = tf.constant([14651])
    one = tf.ones(one_shape, tf.float32)
    pts = tf.stack([X, Y, one], axis=1)
    transform = tf.Variable([[tf.cos(tt),-tf.sin(tt),-xx*tf.cos(tt)+xx+yy*tf.sin(tt)],[tf.sin(tt),tf.cos(tt),-xx*tf.sin(tt)-yy*tf.cos(tt)+yy],[0.0,0.0,1.0]])
    newp = tf.matmul(transform,tf.transpose(pts))
    newp = newp[0:2,:]
    newp = tf.cast(newp, tf.int32)
    ind = tf.transpose(newp)
    new_img = tf.gather_nd(tf_image, ind)
    new_img = tf.reshape(new_img, (1,min_wdw_sz[1],min_wdw_sz[0],1))
    resized_image = tf.image.resize_images(new_img, [100, 100])
    dic = sess.run(resized_image)
    dic = {xxx:resized_image}
    classification_result = sess.run(logits,dic)
    output = tf.argmax(classification_result,1).eval()
    if object_dict[output[0]] == 'mouse':
        prob = classification_result[0][0]/(classification_result[0][0]+classification_result[0][1])
        detections.append((xx, yy, tt, prob))
    return [tf.add(i, 1)]

r = tf.while_loop(while_condition, loop_body, [i])
print(tf_pts[0][0].eval(session=sess))


'''
















