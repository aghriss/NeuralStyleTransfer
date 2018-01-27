# -*- coding: utf-8 -*-
"""
Created on Mon Jan 22 10:09:18 2018

@author: Thinkpad
"""

import tensorflow as tf
import numpy as np

from functools import reduce
from PIL import Image

import skimage.io

def conv_layer(input, weights, bias):
    conv = tf.nn.conv2d(input, tf.constant(weights), strides=(1, 1, 1, 1),padding='SAME')
    return tf.nn.bias_add(conv, bias)
    
def pool_layer(input, pooling):
    if pooling == 'avg':
        return tf.nn.avg_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),padding='SAME')
    else:
        return tf.nn.max_pool(input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1),padding='SAME')

def _tensor_size(tensor):
    from operator import mul
    return reduce(mul, (d.value for d in tensor.get_shape()), 1)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb

def imsave(path,fname, img):
    path2=path
    if path[-1]!="/":
        path2 = path+"/"
    img = np.clip(img, 0, 255).astype(np.uint8)
    Image.fromarray(img).save(path2+fname+".jpg", quality=95)
    
    
def imread(path):
    img = skimage.io.imread(path).astype(np.float)
    if len(img.shape) == 2:
        # grayscale
        img = np.dstack((img,img,img))
    elif img.shape[2] == 4:
        # PNG with alpha channel
        img = img[:,:,:3]
    return img

def merge_images(content, img_out,preserve_colors):
    original_image = np.clip(content, 0, 255)
    styled_image = np.clip(img_out, 0, 255)
    if preserve_colors:      
        # Luminosity transfer steps:
        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
        # 2. Convert stylized grayscale into YUV (YCbCr)
        # 3. Convert original image into YUV (YCbCr)
        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
        # 5. Convert recombined image from YUV back to RGB

        # 1
        styled_grayscale = rgb2gray(styled_image)
        styled_grayscale_rgb = gray2rgb(styled_grayscale)

        # 2
        styled_grayscale_yuv = np.array(Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert('YCbCr'))

        # 3
        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert('YCbCr'))

        # 4
        w, h, _ = original_image.shape
        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
        combined_yuv[..., 1] = original_yuv[..., 1]
        combined_yuv[..., 2] = original_yuv[..., 2]

        # 5
        img_out = np.array(Image.fromarray(combined_yuv, 'YCbCr').convert('RGB'))
    
    return styled_image

def total_variance_loss(image):
    """
        image of shape : [x, height, width, channels]
    """
    TV_loss =(tf.reduce_mean(tf.square(image[:,1:,:,:] - image[:,:-1,:,:]))
            + tf.reduce_mean(tf.square(image[:,:,1:,:] - image[:,:,:-1,:])))
    return TV_loss
