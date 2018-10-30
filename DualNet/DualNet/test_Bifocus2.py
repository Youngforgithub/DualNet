##based on checkpoint/correct4

import os
import time
import numpy as npy
import cv2
import tensorflow as tf
import tensorlayer as tl
import time
from Bifocus_model3 import DualCNN
from data_proc import DataIterSR
from evaluate_metric import sr_metric,merge
from matplotlib import pyplot as plt


def test_SR_DataSet():
    datadir=r"./data/Bifocus"
    img_list=[f for f in os.listdir(datadir) if f.find(".png")!=-1]

    check_point_dir=r"./checkpoint/Bifocus_Correct5"
    res_dir="./test_result/Bifocus2"
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
			##输入图片预处理
            img=cv2.imread(os.path.join(datadir, f), cv2.IMREAD_COLOR)
            [nrow, ncol, nchl]=img.shape
            crop_img=npy.zeros((nrow,ncol,nchl))
            crop_imglr=npy.zeros((nrow,ncol,nchl))
            minSizeX=int(nrow/3)
            minSizeY=int(ncol/3)
            maxSizeX=2*minSizeX
            maxSizeY=2*minSizeY
            crop_img[minSizeX:maxSizeX,minSizeY:maxSizeY,:]=img[minSizeX:maxSizeX,minSizeY:maxSizeY,:]
            crop_img=crop_img/255
            #cv2.imwrite(os.path.join(res_dir, f+"_imgorg.png"),img)
            #cv2.imwrite(os.path.join(res_dir, f+"_imgorg_pt.png"),crop_img)
            img_ds=cv2.resize(img, (minSizeX,minSizeY),interpolation=cv2.INTER_CUBIC)
            img_lr=cv2.resize(img_ds, (ncol, nrow), interpolation=cv2.INTER_CUBIC)
            crop_imglr[minSizeX:maxSizeX,minSizeY:maxSizeY,:]=img_lr[minSizeX:maxSizeX,minSizeY:maxSizeY,:]
            img_out=img_lr-crop_imglr
            img_out=img_out/255
            img_out=img_out[npy.newaxis,:,:,:].astype(npy.float32)
            crop_img=crop_img[npy.newaxis,:,:,:].astype(npy.float32)

			##输入
            start = time.time()
            y_pred,net_D,net_S=sess.run([y_out,netD,netS], feed_dict={x:img_out,crop:crop_img})
            end = time.time()
			##输出处理
            img=(img.astype(npy.float32))/255.0
            img=img[npy.newaxis,:,:,:].astype(npy.float32)
            mse, psnr=sr_metric(img,y_pred)
            mean_time+=end-start
            mean_mse+=mse
            mean_psnr+=psnr
            epoch_cnt+=1
            print("pic_loical,time:{}, mse:{}, psnr:{}".format(end-start,mse, psnr))

            img = cv2.cvtColor(img[0,:,:,:], cv2.COLOR_BGR2RGB)
            y_pred=y_pred[0,:,:,:]/npy.max(y_pred[0,:,:,:])
            y_pred= cv2.cvtColor(y_pred, cv2.COLOR_BGR2RGB)
            net_D=net_D[0,:,:,:]/npy.max(net_D[0,:,:,:])
            net_D= cv2.cvtColor(net_D, cv2.COLOR_BGR2RGB)
            net_S=net_S[0,:,:,:]/npy.max(net_S[0,:,:,:])
            net_S= cv2.cvtColor(net_S, cv2.COLOR_BGR2RGB)
            img_lr= cv2.cvtColor(img_lr, cv2.COLOR_BGR2RGB)
            plt.subplot(2,3,1)
            plt.imshow(img)
            plt.subplot(2,3,2)
            plt.imshow(y_pred)
            plt.subplot(2,3,3)
            plt.imshow(net_D)
            plt.subplot(2,3,4)
            plt.imshow(net_S)
            plt.subplot(2,3,5)
            plt.imshow(img_lr)
            plt.show()
            img_out=(img_out*255).astype(npy.integer)

            for i in range(nrow):
                for j in range(ncol):
                    for k in range(3):
                        if(img_out[0,i,j,k]!=0):
                            net_D[i,j,k]=0
                        if(crop_img[0,i,j,k]!=0):
                            net_S[i,j,k]=0
            plt.subplot(1,2,1)
            plt.imshow(net_D)
            plt.subplot(1,2,2)
            plt.imshow(net_S)
            plt.show()
    print("time:{}, mse:{}, psnr:{}".format(mean_time/epoch_cnt, mean_mse/epoch_cnt, mean_psnr/epoch_cnt))

        
if __name__=="__main__":
    tl.layers.clear_layers_name()
    tf.reset_default_graph() 
    test_SR_DataSet()

