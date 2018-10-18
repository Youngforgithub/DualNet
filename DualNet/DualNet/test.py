##model4 修改了基于model3的基础上，添加了charbonnier滤波器以及ResNet网络结构
import os
import numpy as npy
import tensorflow as tf
import tensorlayer as tl


sess = tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
 
def tensor_expand(tensor_Input,Num):

    tensor_Input = tf.expand_dims(tensor_Input,axis=0)
    tensor_Output = tensor_Input
    for i in range(Num-1):
        tensor_Output= tf.concat([tensor_Output,tensor_Input],axis=0)
    return tensor_Output
 
def get_one_hot_matrix(height,width,position):

    col_length = height
    row_length = width
    col_one_position = position[0]
    row_one_position = position[1]
    rows_num = height
    cols_num = width
 
    single_row_one_hot = tf.one_hot(row_one_position, row_length, dtype=tf.float32)
    single_col_one_hot = tf.one_hot(col_one_position, col_length, dtype=tf.float32)
 
    one_hot_rows = tensor_expand(single_row_one_hot, rows_num)
    one_hot_cols = tensor_expand(single_col_one_hot, cols_num)
    one_hot_cols = tf.transpose(one_hot_cols)
 
    one_hot_matrx = one_hot_rows * one_hot_cols
    return one_hot_matrx
 
def tensor_assign_2D(tensor_input,position,value):
    shape = tensor_input.get_shape().as_list()
    height = shape[0]
    width = shape[1]
    h_index = position[0]
    w_index = position[1]
    one_hot_matrix = get_one_hot_matrix(height, width, position)
 
    new_tensor = tensor_input[:,:] - tensor_input[h_index,w_index]*one_hot_matrix +one_hot_matrix*value
 
    return new_tensor
 
if __name__=="__main__":
    ##test
    tensor_input = tf.constant([i for i in range(60)],tf.float32)
    tensor_crop = tf.constant([2*i for i in range(18)],tf.float32)
    x = tf.reshape(tensor_input,[1,4,5,3])
    crop = tf.reshape(tensor_crop,[1,2,3,3])
    print(x[0,:,:,0].eval())
    print(x[0,:,:,1].eval())
    print(x[0,:,:,2].eval())
    print(crop[0,:,:,0].eval())
    print(crop[0,:,:,1].eval())
    print(crop[0,:,:,2].eval())
    middleX=int(int(x.shape[1])/2)
    middleY=int(int(x.shape[2])/2)
    startX=int(middleX-int(crop.shape[1])/2)
    endX=int(middleX+int(crop.shape[1])/2)
    startY=int(middleY-int(crop.shape[2])/2)
    endY=int(middleY+int(crop.shape[2])/2)
    new_tensor1=x[0,:,:,0]
    new_tensor2=x[0,:,:,1]
    new_tensor3=x[0,:,:,2]
    for i in range(startX,endX):
        for j in range(startY,endY):
            new_tensor1 = tensor_assign_2D(new_tensor1 ,[i,j],crop[0,i-startX,j-startY,0])
            new_tensor2 = tensor_assign_2D(new_tensor2 ,[i,j],crop[0,i-startX,j-startY,1])
            new_tensor3 = tensor_assign_2D(new_tensor3 ,[i,j],crop[0,i-startX,j-startY,2])
    new_tensor1=tf.reshape(new_tensor1,[1,x.shape[1],x.shape[2],1])
    new_tensor2=tf.reshape(new_tensor2,[1,x.shape[1],x.shape[2],1])
    new_tensor3=tf.reshape(new_tensor3,[1,x.shape[1],x.shape[2],1])
    new_tensor=tf.concat([new_tensor1,new_tensor2,new_tensor3],3)
    #new_tensor = tf.reshape(new_tensor,[1,4,5,1])
    print(new_tensor1[0,:,:,0].eval())
    print(new_tensor2[0,:,:,0].eval())
    print(new_tensor3[0,:,:,0].eval())