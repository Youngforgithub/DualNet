import os
import numpy as npy
import tensorflow as tf
import tensorlayer as tl
from Bifocus_model3 import DualCNN
from evaluate_metric import sr_metric,merge
from data_proc import DataIterSR
from matplotlib import pyplot as plt


tl.layers.clear_layers_name()
tf.reset_default_graph()

## charbonnier loss define
def compute_charbonnier_loss(tensor1, tensor2, is_mean=True):
    epsilon = 1e-6
    if is_mean:
        loss = tf.reduce_mean(tf.reduce_mean(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))
    else:
        loss = tf.reduce_mean(tf.reduce_sum(tf.sqrt(tf.square(tf.subtract(tensor1,tensor2))+epsilon), [1, 2, 3]))

    return loss

datadir=r"./data/T91"
img_list=[f for f in os.listdir(datadir) if f.find(".png")!=-1]

scale_factor=4
num_epoch=300000
batch_size=10
train_img_size=42
crop_size=int(train_img_size/3)
lr0=0.0001
print_freq=200
save_freq=5000
check_point_dir="./checkpoint/Bifocus_Correct4"
if not os.path.exists(check_point_dir):
    os.makedirs(check_point_dir)

x=tf.placeholder(tf.float32,shape=[None,train_img_size,train_img_size,3],name="x")
crop=tf.placeholder(tf.float32,shape=[None,train_img_size,train_img_size,3],name="crop")
y=tf.placeholder(tf.float32,shape=[None,train_img_size,train_img_size,3],name="y")

net,netD,netS,endpoints=DualCNN(x,crop)

y_out=net.outputs
y_struct=endpoints["compS"].outputs
y_detail=endpoints["compD"].outputs

cost=compute_charbonnier_loss(y,y_out,is_mean=True)
cost=cost+0.1*compute_charbonnier_loss(y,y_struct, is_mean=True)
cost=cost+0.1*compute_charbonnier_loss(y,y_detail, is_mean=True)



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
                                     feed_dict={x:img_out,crop:img_crop ,y:img_hr})
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
        
        
        
        





