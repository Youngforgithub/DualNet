import tensorflow as tf
import tensorlayer as tl
import cv2

def StructNet(input_data):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    net=tl.layers.Conv2d(input_data,n_filter=64,filter_size=(9,9), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netS_conv0")
    net=tl.layers.Conv2d(net,n_filter=32,filter_size=(1,1),strides=(1,1),
                         act=tf.nn.relu,W_init=W_init,name="netS_conv1")
    net=tl.layers.Conv2d(net,n_filter=3,filter_size=(3,3),strides=(1,1),
                         act=tf.nn.relu,W_init=W_init,name="netS_conv2")
    net=tl.layers.Conv2d(net,n_filter=3,filter_size=(3,3),strides=(1,1),
                         act=tf.nn.relu,W_init=W_init,name="netS_conv3")
    net=tl.layers.MaxPool2d(net,3,3,padding="valid",name="netS_pooling")
    return net
    
def DetailNet(input_data):
     W_init = tf.truncated_normal_initializer(stddev=5e-2)
     net=tl.layers.Conv2d(input_data,n_filter=64,filter_size=(5,5), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netD_conv0")
     for i in range(16):
         out=tl.layers.Conv2d(net,n_filter=32,filter_size=(1,1), strides=(1,1),
                              act=tf.nn.relu,W_init=W_init,name="netD_conv{}".format(i+1))
         out=tl.layers.Conv2d(out,n_filter=64,filter_size=(3,3), strides=(1,1),
                              act=tf.nn.relu,W_init=W_init,name="netD_conv{}".format(i+1))
         net=tl.layers.ConcatLayer([net,out], concat_dim=3, name="netD_concat0")
     out=tl.layers.Conv2d(net,n_filter=32,filter_size=(1,1), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netD_convout1")
     out=tl.layers.Conv2d(out,n_filter=3,filter_size=(3,3), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netD_convout2")
     net=tl.layers.Conv2d(net,n_filter=3,filter_size=(3,3), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netD_convout3")
     net=tl.layers.ConcatLayer([net, out], concat_dim=3,name="netD_concat1")
     net=tl.layers.Conv2d(net,n_filter=3,filter_size=(3,3), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="netD_convout4")
     return net
 
def DualCNN(x,crop):
    W_init = tf.truncated_normal_initializer(stddev=5e-2)
    endpoints={}

    netX=tl.layers.InputLayer(x,name="img_inX")
    netCrop=tl.layers.InputLayer(crop,name="img_inX")
    netD=DetailNet(netCrop)
    netS=StructNet(netX)

    net=tl.layers.ConcatLayer([netD, netS], concat_dim=3,name="net_concat")
    net=tl.layers.DeConv2d(net,n_filter=64,filter_size=(3,3),strides=(3,3),name="net_deconvout")
    net=tl.layers.Conv2d(net,n_filter=3,filter_size=(3,3), strides=(1,1),
                        act=tf.nn.relu,W_init=W_init,name="net_convout1")
    endpoints["compS"]=netS
    endpoints["compD"]=netD
    return net,netD,netS,endpoints
    
    
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
    
  

