import os
import time
import numpy as npy
import cv2
import tensorflow as tf
import tensorlayer as tl
import time
from model4 import DualCNN
from data_proc import DataIterSR

from evaluate_metric import sr_metric


def test_SR_DataSet():
    datadir=r"./data/BSDS200_test"
    img_list=[f for f in os.listdir(datadir) if f.find(".png")!=-1]

    check_point_dir=r"./checkpoint/model4/sr/train_by_bsd200_scale2"
    res_dir="./test_result/model/sr/bsds200_scale2_trained_by_bsds2002"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    x=tf.placeholder(tf.float32,shape=[None,None,None,3],name="x")
    net,netD,netS,endpoints=DualCNN(x)
    y_out=net.outputs

    batch_size=10
    test_img_size=41
    scale_factor=2
    pitch=30

    saver=tf.train.Saver()
    data_iter=DataIterSR(datadir, img_list, batch_size, test_img_size, scale_factor, True)

    with tf.Session() as sess:
        saver.restore(sess,os.path.join(check_point_dir,"model_{}.ckpt".format(95000)))
        mean_time=0
        mean_mse=0
        mean_psnr=0
        epoch_cnt=0
        for f in img_list:
            img=cv2.imread(os.path.join(datadir, f), cv2.IMREAD_COLOR)
            crop_img=img[80:80+pitch,80:80+pitch,:]
            cv2.imwrite(os.path.join(res_dir, f+"_imgorg.png"),img)
            cv2.imwrite(os.path.join(res_dir, f+"_imgorg_pt.png"),crop_img)
            [nrow, ncol, nchl]=img.shape
            img_blur=cv2.GaussianBlur(img,(3,3),1.2)
            img_ds=cv2.resize(img_blur, (ncol//scale_factor, nrow//scale_factor),
                              interpolation=cv2.INTER_CUBIC)
            img_lr=cv2.resize(img_ds, (ncol, nrow), interpolation=cv2.INTER_CUBIC)
            img_in=(img_lr.astype(npy.float32))/255.0
            img_in=img_in[npy.newaxis,:,:,:].astype(npy.float32)
            start = time.time()
            y_pred=sess.run(y_out, feed_dict={x:img_in})
            end = time.time()
            img=(img.astype(npy.float32))/255.0
            img=img[npy.newaxis,:,:,:].astype(npy.float32)
            mse, psnr=sr_metric(img,y_pred)
            mean_time+=end-start
            mean_mse+=mse
            mean_psnr+=psnr
            epoch_cnt+=1
            print("pic_loical,time:{}, mse:{}, psnr:{}".format(end-start,mse, psnr))
            img_out=npy.maximum(0, npy.minimum(1,y_pred[0,:,:,:]))*255
            img_out=img_out.astype(npy.uint8)
            crop_imglr=img_lr[80:80+pitch,80:80+pitch,:]
            crop_imgsr=img_out[80:80+pitch,80:80+pitch,:]
            cv2.imwrite(os.path.join(res_dir, f+"_imglr.png"),img_lr)
            cv2.imwrite(os.path.join(res_dir, f+"_imglr_pt.png"),crop_imglr)
            cv2.imwrite(os.path.join(res_dir, f+str(psnr)+"_imgsr.png"),img_out)
            cv2.imwrite(os.path.join(res_dir, f+str(psnr)+"_imgsr_pt.png"),crop_imgsr)
    print("time:{}, mse:{}, psnr:{}".format(mean_time/epoch_cnt, mean_mse/epoch_cnt, mean_psnr/epoch_cnt))

        
if __name__=="__main__":
    tl.layers.clear_layers_name()
    tf.reset_default_graph() 
    #test_SR()    
    test_SR_DataSet()