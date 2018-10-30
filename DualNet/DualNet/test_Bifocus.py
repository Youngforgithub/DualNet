##based on checkpoint/correct3

import os
import time
import numpy as npy
import cv2
import tensorflow as tf
import tensorlayer as tl
import time
from Bifocus_model2 import DualCNN
from data_proc import DataIterSR
from evaluate_metric import sr_metric,merge
from matplotlib import pyplot as plt


def test_SR_DataSet():
    datadir=r"./data/Bifocus"
    img_list=[f for f in os.listdir(datadir) if f.find(".png")!=-1]

    check_point_dir=r"./checkpoint/Bifocus_Correct5"
    res_dir="./test_result/Bifocus"
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)

    x=tf.placeholder(tf.float32,shape=[None,None,None,3],name="x")
    crop=tf.placeholder(tf.float32,shape=[None,None,None,3],name="crop")
    net,netD,netS,endpoints=DualCNN(x,crop)
    netD=netD.outputs
    netS=netS.outputs
    y_out=net.outputs

    batch_size=10
    test_img_size=41
    scale_factor=4

    saver=tf.train.Saver()
    data_iter=DataIterSR(datadir, img_list, batch_size, test_img_size, scale_factor, True)

    with tf.Session() as sess:
        saver.restore(sess,os.path.join(check_point_dir,"model_{}.ckpt".format(295000)))
        mean_time=0
        mean_mse=0
        mean_psnr=0
        epoch_cnt=0
        for f in img_list:
            img=cv2.imread(os.path.join(datadir, f), cv2.IMREAD_COLOR)
            [nrow, ncol, nchl]=img.shape
            crop_img=npy.zeros((nrow,ncol,nchl))
            crop_imglr=npy.zeros((nrow,ncol,nchl))
            minSizeX=int(nrow/3)
            minSizeY=int(ncol/3)
            maxSizeX=2*minSizeX
            maxSizeY=2*minSizeY
            crop_img[minSizeX:maxSizeX,minSizeY:maxSizeY,:]=img[minSizeX:maxSizeX,minSizeY:maxSizeY,:]
            crop_input=img[minSizeX:maxSizeX,minSizeY:maxSizeY,:]
            crop_input=crop_input[npy.newaxis,:,:,:].astype(npy.float32)/255
            #cv2.imwrite(os.path.join(res_dir, f+"_imgorg.png"),img)
            #cv2.imwrite(os.path.join(res_dir, f+"_imgorg_pt.png"),crop_img)
            img_blur=cv2.GaussianBlur(img,(3,3),1.2)
            img_ds=cv2.resize(img_blur, (ncol//scale_factor, nrow//scale_factor),
                              interpolation=cv2.INTER_CUBIC)
            img_lr=cv2.resize(img_ds, (ncol, nrow), interpolation=cv2.INTER_CUBIC)
            crop_imglr[minSizeX:maxSizeX,minSizeY:maxSizeY,:]=img_lr[minSizeX:maxSizeX,minSizeY:maxSizeY,:]
            img_lr=img_lr-crop_imglr+crop_img
            img_in=(img_lr.astype(npy.float32))/255
            img_in=img_in[npy.newaxis,:,:,:].astype(npy.float32)
            start = time.time()
            y_pred,net_D,net_S=sess.run([y_out,netD,netS], feed_dict={x:img_in,crop:crop_input})
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
            img = cv2.cvtColor(img[0,:,:,:], cv2.COLOR_BGR2RGB)
            y_pred=y_pred[0,:,:,:]/npy.max(y_pred[0,:,:,:])
            y_pred= cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB)
            net_D=net_D[0,:,:,:]/npy.max(net_D[0,:,:,:])
            net_D= cv2.cvtColor(net_D, cv2.COLOR_BGR2RGB)
            net_S=net_S[0,:,:,:]/npy.max(net_S[0,:,:,:])
            net_S= cv2.cvtColor(net_S, cv2.COLOR_BGR2RGB)
            plt.subplot(2,2,1)
            plt.imshow(img)
            plt.subplot(2,2,2)
            plt.imshow(y_pred)
            plt.subplot(2,2,3)
            plt.imshow(net_D)
            plt.subplot(2,2,4)
            plt.imshow(net_S)
            plt.show()
            #cv2.imwrite(os.path.join(res_dir, f+"_imglr.png"),img_lr)
            #cv2.imwrite(os.path.join(res_dir, f+"_imgsr.png"),img_out)
    print("time:{}, mse:{}, psnr:{}".format(mean_time/epoch_cnt, mean_mse/epoch_cnt, mean_psnr/epoch_cnt))

        
if __name__=="__main__":
    tl.layers.clear_layers_name()
    tf.reset_default_graph() 
    test_SR_DataSet()
