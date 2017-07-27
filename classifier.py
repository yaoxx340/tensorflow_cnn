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


object_dict = {0:'mouse',1:'bg'}

w=100
h=100
c=1
ind=0


def read_one_image(im):
    img = transform.resize(im,(w,h))
    return np.asarray(img)

def frange(x, y, jump):
  while x < y:
    yield x
    x += jump

def sliding_window(image, window_size,step_size):    
    pair = np.mgrid[890:1400:step_size[0], 250:730:step_size[1],0.0: 6.28: 5].reshape(3,-1).T
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
        print(xx,yy,tt)
        newim = read_one_image(newim)
        #print(xx,yy,newim.shape)
        yield (xx, yy, tt, newim)


if __name__ == "__main__":
    # Parse the command line arguments
    parser = ap.ArgumentParser()
    parser.add_argument('-i', "--image", help="Path to the test image", required=True)
    parser.add_argument('-d','--downscale', help="Downscale ratio", default=1.25,
            type=int)
    parser.add_argument('-v', '--visualise', help="Visualise the sliding window",
            action="store_true")
    args = vars(parser.parse_args())
    strings=args["image"]
    im = imread(args["image"])
    img = imread(args["image"],as_grey=True)
    strings = strings[-6:-4]
    min_wdw_sz = (91, 161)
    halfx = (min_wdw_sz[0]-1)/2
    halfy = (min_wdw_sz[1]-1)/2
    step_size = (100, 100)
    downscale = args['downscale']
    visualise_det = args['visualise']
    detections = []
    iter_ = sliding_window(img, min_wdw_sz, step_size)
    image_batch = tf.placeholder(dtype=tf.float32, shape=[None, 100,100])
    x_batch = tf.placeholder(dtype=tf.int32, shape=[None, ])
    y_batch = tf.placeholder(dtype=tf.int32, shape=[None, ])
    theta_batch = tf.placeholder(dtype=tf.float32, shape=[None, ])


    #clf = joblib.load(model_path)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.import_meta_graph('/home/yuan/Google Drive/Research/cv/CNN_detector/model.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('/home/yuan/Google Drive/Research/cv/CNN_detector/'))
        graph = tf.get_default_graph()
        xx = graph.get_tensor_by_name("x:0")
        logits = graph.get_tensor_by_name("logits_eval:0")
        while next(iter_)!=None:
            x, y, theta, image = next(iter_)
            #print(images_batch.shape)
            data = np.reshape(image, (1,100,100,1))
            feed_dict = {xx:data}
            classification_result = sess.run(logits,feed_dict)
            output = tf.argmax(classification_result,1).eval()
            if object_dict[output[0]] == 'mouse':
                prob = classification_result[0][0]/(classification_result[0][0]+classification_result[0][1])
                detections.append((x, y, theta, prob))

        '''
     
        # Downscale the image and iterate
        for im_scaled in pyramid_gaussian(img, downscale=downscale):
            # This list contains detections at the current scale
            cd = []
            if im_scaled.shape[0] < min_wdw_sz[1] or im_scaled.shape[1] < min_wdw_sz[0]:
                break
            for (x, y, theta, im_window) in sliding_window(im_scaled, min_wdw_sz, step_size):
                #cv2.imshow("window",im_window)
                #cv2.imwrite(str(x)+str(y)+str(theta)+".png", im_window*255)
                #cv2.waitKey(100)
                if im_window.shape[0] != min_wdw_sz[1] or im_window.shape[1] != min_wdw_sz[0]:
                    continue
                # Calculate the HOG features
                data = read_one_image(im_window)
                data=np.reshape(data,(1,100,100,1))
                feed_dict = {xx:data}
                classification_result = sess.run(logits,feed_dict)
                output = tf.argmax(classification_result,1).eval()
                if object_dict[output[0]] == 'mouse':
                    prob = classification_result[0][0]/(classification_result[0][0]+classification_result[0][1])
                    detections.append((x, y, theta, prob,int(min_wdw_sz[0]*(downscale**scale)),int(min_wdw_sz[1]*(downscale**scale))))
                    cd.append(detections[-1])
   
            scale+=1
            break
            '''


    clone = im.copy()
    for (x_tl, y_tl, theta1,_) in detections:
        # Draw the detections
        cv2.rectangle(im, (x_tl-halfx, y_tl-halfy), (x_tl+halfx+1, y_tl+halfy+1), (0, 0, 0), thickness=2)
    #cv2.imshow("Raw Detections before NMS", im)
    cv2.imwrite("/home/yuan/Google Drive/Research/cv/CNN_detector/result/"+str(strings)+"raw.png",im)
    #cv2.waitKey(100)


    detections = nms(detections)

    # Display the results after performing NMS
    for (x_tl, y_tl, theta,_) in detections:
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
        # Draw the detections
        print(theta)
        cv2.polylines(clone,[pts1],True,(0,255,0))
    #cv2.imshow("Final Detections after applying NMS", clone)
    cv2.imwrite("/home/yuan/Google Drive/Research/cv/CNN_detector/result/"+str(strings)+"final.png",clone)
    #cv2.waitKey()
