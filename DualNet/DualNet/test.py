##model4 修改了基于model3的基础上，添加了charbonnier滤波器以及ResNet网络结构
import os
import numpy as npy
import tensorflow as tf
import tensorlayer as tl


A=npy.array([[1,2,3,4],[5,6,7,8]])
print(A)
print(npy.argmax(A))