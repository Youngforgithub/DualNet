import os
import cv2
import random
import tensorflow as tf
import numpy as npy
from matplotlib import pyplot as plt
from evaluate_metric import sr_metric,sr_single
  
class DataIterSR(object):
    def __init__(self, datadir,img_list, crop_num, crop_size, scale_factor, is_shuffle):
        self._datadir=datadir
        self._img_list=img_list
        self._crop_num=crop_num
        self._crop_size=crop_size
        self._scale_fator=scale_factor
        self._is_shuffle=is_shuffle
        self._provide_input=zip(["img_in"],[(crop_num,3, crop_size, crop_size)])
        self._provide_output=zip(["img_out"],[(crop_num,3, crop_size, crop_size)])
        self._num_img=len(img_list)
        self._cur_idx=0
        self._iter_cnt=0
        
    def reset(self):
        self._cur_idx=0
        self._iter_cnt=0
        
    def fetch_next(self):
        if self._is_shuffle and npy.mod(self._cur_idx,self._num_img)==0:
            self._cur_idx=0
            random.shuffle(self._img_list)  
        crop_size=self._crop_size
        img_path=os.path.join(self._datadir,self._img_list[self._cur_idx])
        img=cv2.imread(img_path, cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        [nrow, ncol, nchl]=img.shape
        self._iter_cnt += 1
        self._cur_idx += 1
        if nrow < crop_size or ncol < crop_size:
            raise ValueError("Crop size is larger than image size")
        img_blur=cv2.GaussianBlur(img,(3,3),1.2)
        img_struct=cv2.GaussianBlur(img_blur,(3,3),1.5)
        img_ds=cv2.resize(img_blur, (ncol//self._scale_fator, nrow//self._scale_fator),
                          interpolation=cv2.INTER_CUBIC)
        img_lr=cv2.resize(img_ds, (ncol, nrow), interpolation=cv2.INTER_CUBIC)
        img=img.astype(npy.float32)
        img_lr=img_lr.astype(npy.float32)
        img_struct=img_struct.astype(npy.float32)
        img_detail=img-img_struct
        sub_img_hr=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        sub_img_lr=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        sub_img_struct=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        sub_img_detail=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        for i in range(self._crop_num):
            nrow_start=npy.random.randint(0,nrow-crop_size)
            ncol_start=npy.random.randint(0,ncol-crop_size)
            img_crop=img_lr[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]              
            img_crop=img_crop/255.0       
            sub_img_lr[i,:,:,:]=img_crop
            
            img_crop=img[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]
            img_crop=img_crop/255.0              
            sub_img_hr[i,:,:,:]=img_crop

            img_crop=img_struct[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]
            img_crop=img_crop/255.0             
            sub_img_struct[i,:,:,:]=img_crop

            img_crop=img_detail[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]
            img_crop=img_crop/255.0                   
            sub_img_detail[i,:,:,:]=img_crop

        return (sub_img_hr.astype(npy.float32),sub_img_lr.astype(npy.float32),
                sub_img_struct.astype(npy.float32),sub_img_detail.astype(npy.float32))
    def fetch_next2(self):
        if self._is_shuffle and npy.mod(self._cur_idx,self._num_img)==0:
            self._cur_idx=0
            random.shuffle(self._img_list)  
        crop_size=self._crop_size
        img_path=os.path.join(self._datadir,self._img_list[self._cur_idx])
        img=cv2.imread(img_path, cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        [nrow, ncol, nchl]=img.shape
        self._iter_cnt += 1
        self._cur_idx += 1
        if nrow < crop_size or ncol < crop_size:
            raise ValueError("Crop size is larger than image size")
        img_blur=cv2.GaussianBlur(img,(3,3),1.2)
        gray=cv2.cvtColor(img_blur,cv2.COLOR_RGB2GRAY)
        xgrad=cv2.Sobel(gray,cv2.CV_16SC1,1,0)
        ygrad=cv2.Sobel(gray,cv2.CV_16SC1,0,1)
        img_detail=cv2.Canny(xgrad,ygrad,50,150)
        img_detail=cv2.bitwise_and(img,img,mask=img_detail)
        #img_struct=cv2.GaussianBlur(img_blur,(3,3),1.5)
        img_ds=cv2.resize(img_blur, (ncol//self._scale_fator, nrow//self._scale_fator),
                          interpolation=cv2.INTER_CUBIC)
        img_lr=cv2.resize(img_ds, (ncol, nrow), interpolation=cv2.INTER_CUBIC)
        img=img.astype(npy.float32)
        img_lr=img_lr.astype(npy.float32)
        img_struct=img-img_detail
        img_struct=img_struct.astype(npy.float32)
        
        sub_img_hr=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        sub_img_lr=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        sub_img_struct=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        sub_img_detail=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        for i in range(self._crop_num):
            nrow_start=npy.random.randint(0,nrow-crop_size)
            ncol_start=npy.random.randint(0,ncol-crop_size)
            img_crop=img_lr[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]              
            img_crop=img_crop/255.0       
            sub_img_lr[i,:,:,:]=img_crop
            
            img_crop=img[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]
            img_crop=img_crop/255.0              
            sub_img_hr[i,:,:,:]=img_crop

            img_crop=img_struct[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]
            img_crop=img_crop/255.0             
            sub_img_struct[i,:,:,:]=img_crop

            img_crop=img_detail[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]
            img_crop=img_crop/255.0                   
            sub_img_detail[i,:,:,:]=img_crop
        #plt.figure()
        #plt.imshow(img)
        #plt.figure()
        #plt.show(sub_img_lr.astype(npy.float32)[0,:,:,:])
        #plt.show()
        return (sub_img_hr.astype(npy.float32),sub_img_lr.astype(npy.float32),
                sub_img_struct.astype(npy.float32),sub_img_detail.astype(npy.float32))

    def creat(self):
        if self._is_shuffle and npy.mod(self._cur_idx,self._num_img)==0:
            self._cur_idx=0
            random.shuffle(self._img_list)  
        crop_size=self._crop_size
        img_path=os.path.join(self._datadir,self._img_list[self._cur_idx])
        img=cv2.imread(img_path, cv2.IMREAD_COLOR)
        img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        [nrow, ncol, nchl]=img.shape
        img_ds=cv2.resize(img, (ncol//self._scale_fator, nrow//self._scale_fator),
                          interpolation=cv2.INTER_CUBIC)
     
        img_lr=cv2.resize(img_ds, (ncol, nrow), interpolation=cv2.INTER_CUBIC)
        img_lr=img_lr/255
        img_lr=img_lr.astype(npy.float32)

        pixel_size=crop_size
        min_size=int(crop_size/3)
        max_size=int(crop_size/3)*2
        crop_size=int(crop_size/3)

        img_hr=npy.zeros((1,pixel_size,pixel_size,3))
        img_lw=npy.zeros((1,pixel_size,pixel_size,3))
        img_crop_lr=npy.zeros((1,pixel_size,pixel_size,3))
        img_crop=npy.zeros((1, pixel_size, pixel_size,3))
        img_rs=npy.zeros((1, pixel_size, pixel_size, 3))
        sub_imgCrop=npy.zeros((1, crop_size, crop_size, 3))
        for i in range(1):
            nrow_start=npy.random.randint(0,nrow-pixel_size)
            ncol_start=npy.random.randint(0,ncol-pixel_size)
            img_hr[i,:,:,:]=img[nrow_start:nrow_start+pixel_size,ncol_start:ncol_start+pixel_size,:]/255

            img_crop[i,min_size:max_size,
					   min_size:max_size,:]=img[nrow_start+min_size:nrow_start+max_size,
												ncol_start+min_size:ncol_start+max_size,:]
            sub_imgCrop[i,:,:,:]=img[nrow_start+min_size:nrow_start+max_size,
												ncol_start+min_size:ncol_start+max_size,:]
            img_crop_lr[i,min_size:max_size,
					   min_size:max_size,:]=img_lr[nrow_start+min_size:nrow_start+max_size,
												   ncol_start+min_size:ncol_start+max_size,:]
            #sub_img_lr[i,:,:,:]=img_crop
            #img_crop=img[nrow_start:nrow_start+crop_size,
            #                ncol_start:ncol_start+crop_size,:]
            img_crop[i,:,:,:]=img_crop[i,:,:,:]/255  
            img_lw[i,:,:,:]=img_lr[nrow_start:nrow_start+pixel_size,ncol_start:ncol_start+pixel_size,:]
            img_rs[i,:,:,:]=img_lw[i,:,:,:]-img_crop_lr[i,:,:,:]+img_crop[i,:,:,:]


        return (img_hr.astype(npy.float32),img_lw.astype(npy.float32),
				img_rs.astype(npy.float32),(sub_imgCrop/255).astype(npy.float32))



class DataIterEPF(object):
    def __init__(self, datadir,img_list, crop_num, crop_size, is_shuffle):
        self._datadir=datadir
        self._img_list=img_list
        self._crop_num=crop_num
        self._crop_size=crop_size
        self._is_shuffle=is_shuffle
        self._provide_input=zip(["img_in"],[(crop_num,3, crop_size, crop_size)])
        self._provide_output=zip(["img_out"],[(crop_num,3, crop_size, crop_size)])
        self._num_img=len(img_list)
        self._cur_idx=0
        self._iter_cnt=0
        
    def reset(self):
        self._cur_idx=0
        self._iter_cnt=0
        
    def fetch_next(self):
        if self._is_shuffle and npy.mod(self._cur_idx,self._num_img)==0:
            self._cur_idx=0
            random.shuffle(self._img_list) 
        crop_size=self._crop_size
        img_path1=os.path.join(self._datadir,self._img_list[self._cur_idx][0])
        img1=cv2.imread(img_path1, cv2.IMREAD_COLOR)
        img1=cv2.cvtColor(img1,cv2.COLOR_BGR2RGB)
        [nrow1, ncol1, nchl1]=img1.shape
        img_path2=os.path.join(self._datadir,self._img_list[self._cur_idx][1])
        img2=cv2.imread(img_path2, cv2.IMREAD_COLOR)
        img2=cv2.cvtColor(img2,cv2.COLOR_BGR2RGB)
        [nrow, ncol, nchl]=img2.shape
        
        if (nrow1!=nrow) or ncol1!=ncol or nchl1 !=nchl:
            raise ValueError("Two images have different size")
     
        self._iter_cnt += 1
        self._cur_idx += 1
        if nrow < crop_size or ncol < crop_size:
            raise ValueError("Crop size is larger than image size")
        img1=img1.astype(npy.float32)
        img2=img2.astype(npy.float32)     
        sub_img1=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        sub_img2=npy.zeros((self._crop_num, crop_size, crop_size, 3))
        for i in range(self._crop_num):
            nrow_start=npy.random.randint(0,nrow-crop_size)
            ncol_start=npy.random.randint(0,ncol-crop_size)
            img_crop=img1[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]              
            img_crop=img_crop/255.0       
            sub_img1[i,:,:,:]=img_crop
            
            img_crop=img2[nrow_start:nrow_start+crop_size,
                            ncol_start:ncol_start+crop_size,:]
            img_crop=img_crop/255.0              
            sub_img2[i,:,:,:]=img_crop

        return (sub_img1.astype(npy.float32),sub_img2.astype(npy.float32))    
 
def test_SRDataIter():
    datadir=r"data\T91"
    img_list=[f for f in os.listdir(datadir) if f.find(".png")!=-1]
    crop_num=5
    crop_size=100
    scale_factor=3
    data_iter=DataIterSR(datadir, img_list, crop_num, crop_size, scale_factor, True)
    res_dir='./test_result/outsave'
    if not os.path.exists(res_dir):
        os.makedirs(res_dir)
    try:
        img_hr, img_lr, img_rs,img_crop=data_iter.creat()
        plt.subplot(2,2,1)
        plt.imshow(img_hr[0,:,:,:])
        plt.title('origin image')
        plt.axis('off')
        img_hr = cv2.cvtColor(img_hr[0,:,:,:], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(res_dir,"_imghr.png"),img_hr*255)
        plt.subplot(2,2,2)
        plt.imshow(img_lr[0,:,:,:])
        plt.title('low level image')
        plt.axis('off')
        img_lr = cv2.cvtColor(img_lr[0,:,:,:], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(res_dir,"_imglr.png"),img_lr*255)
        plt.subplot(2,2,3)
        plt.imshow(img_rs[0,:,:,:])
        plt.title('reconstruct image 0')
        plt.axis('off')
        img_rs = cv2.cvtColor(img_rs[0,:,:,:], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(res_dir,"_imgrs.png"),img_rs*255)
        plt.subplot(2,2,4)
        plt.imshow(img_crop[0,:,:,:])
        plt.title('crop image 0')
        plt.axis('off')
        img_crop = cv2.cvtColor(img_crop[0,:,:,:], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(res_dir,"_imgcrop.png"),img_crop*255)
        plt.show()
        mse, psnr=sr_single(img_hr, img_rs)
        print("mse={}, psnr={}".format(mse, psnr))
    except ValueError:
        print("data_iter get no data")
        
def test_DataIterEPF():
    datadir=r"data\T91"
    img_list1=[f for f in os.listdir(datadir) if f.find(".png")!=-1 and f.find("norain")!=-1]
    img_list2=[f for f in os.listdir(datadir) if f.find(".png")!=-1 
               and f.find("rain")!=-1 and f.find("norain")==-1]
    img_list=[[f1,f2] for f1, f2 in zip(img_list1,img_list2)]
    crop_num=5
    crop_size=64
    data_iter=DataIterEPF(datadir, img_list, crop_num, crop_size, True)
    try:
        img_norain, img_rain=data_iter.fetch_next()
        plt.figure()
        plt.subplot(1,2,1)
        plt.imshow(img_norain[0, :,:,:])
        plt.subplot(1,2,2)
        plt.imshow(img_rain[0, :,:,:])
        mse, psnr=sr_metric(img_norain, img_rain)
        print("mse={}, psnr={}".format(mse, psnr))
    except ValueError:
        print("data_iter get no data")

if __name__=="__main__":
    test_SRDataIter()
    #test_DataIterEPF()
              