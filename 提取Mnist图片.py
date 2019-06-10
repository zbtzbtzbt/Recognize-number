#!/usr/bin/env python
#-*- coding:utf-8 -*-
# author:zbt
# datetime:2019/6/5 23:47
# software: PyCharm
import numpy as np
import struct
import matplotlib.pyplot as plt
from PIL import Image,ImageFont,ImageDraw
import scipy.misc

filename = 'MNIST_data\\train-images.idx3-ubyte'
filename1 = 'MNIST_data\\train-labels.idx1-ubyte'

binfile = open(filename,'rb')#以二进制方式打开
lbinfile = open(filename1,'rb')
buf  = binfile.read()
lbuf = lbinfile.read()

index = 0
lind  = 0
magic, numImages, numRows, numColums = struct.unpack_from('>IIII',buf,index)#读取4个32 int
print (magic,' ',numImages,' ',numRows,' ',numColums  )
index += struct.calcsize('>IIII')

lmagic, numl = struct.unpack_from('>II',lbuf,lind)
print ('label')
print (lmagic,' ', numl)
lind += struct.calcsize('>II')

outputLabel='MNIST_data\\labels.txt'
fw=open(outputLabel,"w+")

outputImgDir='MNIST_data\\'

for i in range(numl):
    im = struct.unpack_from('>784B',buf,index)
    index += struct.calcsize('>784B' )
    im = np.array(im)

    #np.transpose(im)
    #print im.shape

    im = im.reshape(28,28)
    imgdir=outputImgDir+str(i)+'.jpg'
    scipy.misc.imsave(imgdir, im)

##########3
    #tlabel=np.array((struct.unpack_from('>1B',lbuf,lind)))[0]
    tlabel=np.array((struct.unpack_from('>1B',lbuf,lind)))[0]
    #print tlabel
    fw.write(str(tlabel)+"\n")
    lind+=struct.calcsize('>1B')

fw.close()
binfile.close()
lbinfile.close()