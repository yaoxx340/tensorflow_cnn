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
    #r = image[:,:,0]
    #g = image[:,:,1]
    #b = image[:,:,2]

    halfx = (window_size[0]-1)/2
    halfy = (window_size[1]-1)/2
    for y in xrange(250, 730, step_size[1]):
        for x in xrange(890, 1400, step_size[0]):
            lstx = range(x-halfx, x+halfx+1)
            lsty = range(y-halfy, y+halfy+1)
            pts = []
            for j in lsty:
                for i in lstx:
                    pts.append([i,j,1])
            pts = np.transpose(pts)
            for theta in frange(0, 6.28, 0.174):
                transform = np.array([[np.cos(theta),-np.sin(theta),-x*np.cos(theta)+x+y*np.sin(theta)],[np.sin(theta),np.cos(theta),-x*np.sin(theta)-y*np.cos(theta)+y],[0,0,1]])
                newp = fast_dot(transform,pts)
                newp = newp[0:2,:]
                newp = np.transpose(newp).astype(int).tolist()
                new = []
                #newr = []
                #newg = []
                #newb = []
                for i in range(0,len(newp)):
                    indx = newp[i][1]
                    indy = newp[i][0]
                    new.append(image[indx,indy])
                    #newr.append(r[indx,indy])
                    #newg.append(g[indx,indy])
                    #newb.append(b[indx,indy])
                #newrr=np.array(newr).reshape((window_size[1],window_size[0]))
                #newgg=np.array(newg).reshape((window_size[1],window_size[0]))
                #newbb=np.array(newb).reshape((window_size[1],window_size[0]))
                #newim = np.dstack((newrr,newgg,newbb))
                newim=np.array(new).reshape((window_size[1],window_size[0]))
                yield (x, y, theta, newim)


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
    step_size = (10, 10)
    downscale = args['downscale']
    visualise_det = args['visualise']
    detections = []
    #clf = joblib.load(model_path)
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        saver = tf.train.import_meta_graph('/home/yaoxx340/CNN_detector/model.ckpt.meta')
        saver.restore(sess,tf.train.latest_checkpoint('/home/yaoxx340/CNN_detector/'))
        graph = tf.get_default_graph()
        xx = graph.get_tensor_by_name("x:0")
        logits = graph.get_tensor_by_name("logits_eval:0")

        scale = 0
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
                    #print ("Detection:: Location -> ({}, {})".format(x, y))
                    #print ("Scale ->  {} | Confidence Score {} \n".format(scale,prob))
                    detections.append((x, y, theta, prob,int(min_wdw_sz[0]*(downscale**scale)),int(min_wdw_sz[1]*(downscale**scale))))
                    cd.append(detections[-1])
                # If visualize is set to true, display the working
                # of the sliding window
                #if visualise_det:
                    #clone = im_scaled.copy()
                    #for x1, y1,theta1,_, _, _  in cd:
                        # Draw the detections at this scale
                        #cv2.rectangle(clone, (x1-halfx, y1-halfy), (x1 + halfx+1, y1 +halfy+1), (0, 0, 0), thickness=2)
                    #cv2.rectangle(clone, (x-halfx, y-halfy), (x + halfx+1, y +halfy+1), (255, 255, 255), thickness=2)
                    #cv2.imshow("Sliding Window in Progress", clone)
                    #cv2.waitKey(10)
            # Move the the next scale
            scale+=1
            break


    clone = im.copy()
    for (x_tl, y_tl, theta1,_, w, h) in detections:
        # Draw the detections
        cv2.rectangle(im, (x_tl-halfx, y_tl-halfy), (x_tl+halfx+1, y_tl+halfy+1), (0, 0, 0), thickness=2)
    #cv2.imshow("Raw Detections before NMS", im)
    cv2.imwrite("/home/yaoxx340/CNN_detector/result/"+str(strings)+"raw.png",im)
    #cv2.waitKey(100)


    detections = nms(detections)

    # Display the results after performing NMS
    for (x_tl, y_tl, theta,_, ww, hh) in detections:
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
    cv2.imwrite("/home/yaoxx340/CNN_detector/result/"+str(strings)+"final.png",clone)
    #cv2.waitKey()
