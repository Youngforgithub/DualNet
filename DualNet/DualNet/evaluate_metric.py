import numpy as npy
import tensorflow as tf
#import cv2

def sr_metric(hr_imgs, sr_imgs):
    sr_shape=sr_imgs.shape
    hr_shape=hr_imgs.shape
    rs=(hr_shape[1]-sr_shape[1])//2
    cs=(hr_shape[2]-sr_shape[2])//2
    hr_imgs=hr_imgs*255
    sr_imgs=sr_imgs*255
    
    hr_center_imgs=hr_imgs[:,rs:rs+sr_shape[1],cs:cs+sr_shape[2],:]

    psnr=npy.zeros(hr_shape[0],dtype=npy.float32)
    mse=npy.zeros(hr_shape[0],dtype=npy.float32)
    for i in range(hr_shape[0]):
        diff=hr_center_imgs[i,:,:,:]-sr_imgs[i,:,:,:]       
        mse[i]=npy.mean(diff*diff)
        psnr[i]=10*npy.log10(255*255/mse[i])
#    print psnr
    return (npy.mean(npy.sqrt(mse)),npy.mean(psnr))
    
def sr_single(hr_imgs, sr_imgs):
    sr_shape=sr_imgs.shape
    hr_shape=hr_imgs.shape
    rs=(hr_shape[0]-sr_shape[0])//2
    cs=(hr_shape[1]-sr_shape[1])//2
    hr_imgs=hr_imgs*255
    sr_imgs=sr_imgs*255
    
    hr_center_imgs=hr_imgs[rs:rs+sr_shape[0],cs:cs+sr_shape[1],:]

    psnr=npy.zeros(1,dtype=npy.float32)
    mse=npy.zeros(1,dtype=npy.float32)
    diff=hr_center_imgs[:,:,:]-sr_imgs[:,:,:]       
    mse=npy.mean(diff*diff)
    psnr=10*npy.log10(255*255/mse)
#    print psnr
    return (npy.sqrt(mse),psnr)



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
 
    new_tensor = tensor_input - tensor_input[h_index,w_index]*one_hot_matrix +one_hot_matrix*value
 
    return new_tensor




def merge(x,crop):
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
    return new_tensor
    

        

