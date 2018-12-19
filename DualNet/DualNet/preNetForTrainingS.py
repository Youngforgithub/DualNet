import tensorflow as tf
import tensorlayer as tl
from numpy  import *
import os
from data_proc import DataIterSR
from PIL import Image
import numpy as npy
from evaluate_metric import sr_metric,merge


def compute_charbonnier_loss(tensor1, tensor2, is_mean=True):
    epsilon = 1e-6
    if is_mean:
        loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))
    else:
        loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))
    return loss

def netD(input_data):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    net=tl.layers.InputLayer(input_data,name="out_in")
    net1=tl.layers.Conv2d(net,n_filter=64,filter_size=(3,3), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netS_conv0")
    net=tl.layers.Conv2d(net1,n_filter=128,filter_size=(3,3),strides=(1,1),
                         act=tf.nn.relu,W_init=W_init,name="netS_conv1")
    net=tl.layers.Conv2d(net,n_filter=256,filter_size=(3,3),strides=(1,1),
                         act=tf.nn.relu,W_init=W_init,name="netS_conv2")
    net=tl.layers.ConcatLayer([net1,net],concat_dim=3,name="netS_concat")
    net=tl.layers.Conv2d(net,n_filter=512,filter_size=(3,3),strides=(1,1),
						 act=tf.nn.relu,W_init=W_init,name="netS_conv3")
    net=tl.layers.Conv2d(net,n_filter=128,filter_size=(3,3),strides=(1,1),
						 act=tf.nn.relu,W_init=W_init,name="netS_conv4")
    net=tl.layers.Conv2d(net,n_filter=64,filter_size=(3,3),strides=(1,1),
						 act=tf.nn.relu,W_init=W_init,name="netS_conv5")
    net=tl.layers.Conv2d(net,n_filter=3,filter_size=(3,3),strides=(1,1),
						 act=tf.nn.relu,W_init=W_init,name="netS_conv6")
    return net


datadir=r"./data/BSDS200"
img_list=[f for f in os.listdir(datadir) if f.find(".png")!=-1]

scale_factor=2
num_epoch=300000
batch_size=10
train_img_size=42
crop_size=42
lr0=0.0001
print_freq=200
save_freq=5000
check_point_dir="./checkpoint/preTrainD"
if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)

# # 定义数据，设置占位符
crop=tf.placeholder(tf.float32,shape=[None,crop_size,crop_size,3],name="crop")
y=tf.placeholder(tf.float32,shape=[None,crop_size,crop_size,3],name="y")

net=netD(crop)
y_out=net.outputs
cost=compute_charbonnier_loss(y,y_out,is_mean=True)
global_step = tf.Variable(0)
lr = tf.train.exponential_decay(lr0, global_step, 100, 0.96, staircase=True) 
train_op=tf.train.AdamOptimizer(lr).minimize(cost)

data_iter=DataIterSR(datadir, img_list, batch_size, train_img_size, scale_factor, True)
saver=tf.train.Saver()

with tf.Session() as sess:
    tl.layers.initialize_global_variables(sess)
    net.print_params()
    net.print_layers()
    mean_loss=0
    mean_mse=0
    mean_psnr=0
    epoch_cnt=0
    for epoch in range(num_epoch):
        img_hr, img_lr, img_out, img_crop=data_iter.creat2()
        train_loss,y_pred,_=sess.run([cost,y_out,train_op],
                                     feed_dict={crop:img_lr ,y:img_hr})
        mse, psnr=sr_metric(img_hr,y_pred)
        mean_loss+=train_loss
        mean_mse+=mse
        mean_psnr+=psnr
        epoch_cnt+=1
        if npy.mod(epoch,print_freq)==0:
            print("Epoch:{},train_loss:{}, mse:{}, psnr:{}".format(
                  epoch, mean_loss/epoch_cnt,mean_mse/epoch_cnt, mean_psnr/epoch_cnt))
            mean_loss=0
            mean_mse=0
            mean_psnr=0
            epoch_cnt=0
        if epoch>0 and npy.mod(epoch,save_freq)==0:
            print("Saving model at epoch {}".format(epoch))
            saver.save(sess,os.path.join(check_point_dir,"model_{}.ckpt".format(epoch)))
