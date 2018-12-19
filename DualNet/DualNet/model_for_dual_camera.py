import tensorflow as tf
import tensorlayer as tl
from patch_match import patch_match
import tensorflow.contrib.slim as slim
import os



def load_ckpt():
    checkpointD = "./checkpoint/preTrainD"
    sess = tf.Session()
    excludeD = ['netS_concat','netD_convout2','netD_convout3','netD_convout1']
    for i in range(4):
        excludeD.append('netS_conv{}'.format(i))
    variables_to_restoreD = slim.get_variables_to_restore(exclude=excludeD)
    saver = tf.train.Saver(variables_to_restoreD)
    saver.restore(sess, os.path.join(checkpointD,'model_295000.ckpt'))

    checkpointS = "./checkpoint/preTrainS"
    sess = tf.Session()
    excludeS = ['netS_conv4','netS_conv5','netS_conv6','netD_conv0']
    for i in range(4):
        excludeS.append('netD_convout{}'.format(i))
    variables_to_restoreS = slim.get_variables_to_restore(exclude=excludeS)
    saver = tf.train.Saver(variables_to_restoreS)
    saver.restore(sess, os.path.join(checkpointS,'model_295000.ckpt'))

def StructNet(input_data):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    net1=tl.layers.Conv2d(input_data,n_filter=64,filter_size=(3,3), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netS_conv0")
    net=tl.layers.Conv2d(net1,n_filter=128,filter_size=(3,3),strides=(1,1),
                         act=tf.nn.relu,W_init=W_init,name="netS_conv1")
    net=tl.layers.Conv2d(net,n_filter=256,filter_size=(3,3),strides=(1,1),
                         act=tf.nn.relu,W_init=W_init,name="netS_conv2")
    net=tl.layers.ConcatLayer([net1,net],concat_dim=3,name="netS_concat")
    net=tl.layers.Conv2d(net,n_filter=512,filter_size=(3,3),strides=(1,1),
						 act=tf.nn.relu,W_init=W_init,name="netS_conv3")
    return net
    
def DetailNet(input_data):
     W_init = tf.truncated_normal_initializer(stddev=5e-2)
     net=tl.layers.Conv2d(input_data,n_filter=64,filter_size=(3,3), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netD_conv0")
     net=tl.layers.Conv2d(net,n_filter=128,filter_size=(1,1), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netD_convout0")
     net=tl.layers.Conv2d(net,n_filter=512,filter_size=(3,3), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netD_convout1")
     return net
 
def DualCNN(x,crop):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    endpoints={}
    net=tl.layers.InputLayer(x,name="out_in")
    crop=tl.layers.InputLayer(crop,name="crop_in")
    netD=DetailNet(crop)
    netS=StructNet(net)
    load_ckpt()
    endpoints["compS"]=netS
    endpoints["compD"]=netD
    return netS,netD,endpoints

def PatchNet(input_data,patch_data):
     W_init = tf.truncated_normal_initializer(stddev=5e-2)
     net=tl.layers.ConcatLayer([input_data,patch_data],concat_dim=3,name="netP_concat0")
     net=tl.layers.Conv2d(net,n_filter=1024,filter_size=(3,3), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netP_conv0")
     for i in range(16):
         net_in=tl.layers.Conv2d(net,n_filter=64,filter_size=(1,1), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netP_iner1_conv{}".format(i+1))
         net_in=tl.layers.Conv2d(net_in,n_filter=512,filter_size=(1,1), strides=(1,1),
                              act=tf.nn.relu,W_init=W_init,name="netP_iner2_conv{}".format(i+1))
         net_in=tl.layers.Conv2d(net_in,n_filter=1024,filter_size=(3,3), strides=(1,1),
                              act=tf.nn.relu,W_init=W_init,name="netD_iner3_conv{}".format(i+1))
         net=tl.layers.ConcatLayer([net,net_in],concat_dim=3,name="netP_concat{}".format(i+1))
     net=tl.layers.Conv2d(net,n_filter=512,filter_size=(1,1), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netP_convout0")
     net=tl.layers.Conv2d(net,n_filter=128,filter_size=(3,3), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netP_convout1")
     net=tl.layers.Conv2d(net,n_filter=3,filter_size=(3,3), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netP_convout2")
     return net

def PatchCNN(x,patch):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    endpoints={}
    net=tl.layers.InputLayer(x,name="net_in")
    crop=tl.layers.InputLayer(patch,name="patch_in")
    net=PatchNet(net,crop)

    return net
    
if __name__=="__main__":
    tl.layers.clear_layers_name()
    tf.reset_default_graph()
    sess = tf.InteractiveSession()
    x=tf.placeholder(tf.float32,shape=[None,64,64,3],name="x")
    y=tf.placeholder(tf.float32,shape=[None,64,64,3],name="y")
    net,endpoints=DualCNN(x)
    tl.layers.initialize_global_variables(sess)
#    print(tf.shape(net))
    net.print_params()
    net.print_layers()
    
  

