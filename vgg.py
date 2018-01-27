# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io
import config
import utils


class VGG_NET(object):
    
    def __del__(self):
        print('VGG network deleted')

    def __init__(self):

        self.data = scipy.io.loadmat(config.VGG_PATH)
        if not all(i in self.data for i in ('layers', 'classes', 'normalization')):
            raise ValueError("You're using the wrong VGG19 data. Please follow the instructions in the README to download the correct data.")
            self.__del__()
            
        self.mean_pixel = np.mean(self.data['normalization'][0][0][0], axis=(0, 1))
        self.weights = self.data['layers'][0]
        
    def preload(self,input_image, pooling):
        net = {}
        current = input_image
        for i, name in enumerate(config.VGG19_LAYERS):
            kind = name[:4]
            if kind == 'conv':
                kernels, bias = self.weights[i][0][0][0][0]
                # matconvnet: weights are [width, height, in_channels, out_channels]
                # tensorflow: weights are [height, width, in_channels, out_channels]
                kernels = np.transpose(kernels, (1, 0, 2, 3))
                bias = bias.reshape(-1)
                current = utils.conv_layer(input_image, kernels, bias)
            elif kind == 'relu':
                current = tf.nn.relu(input_image)
            elif kind == 'pool':
                current = utils.pool_layer(input_image, pooling)
            net[name] = current
        assert len(net) == len(config.VGG19_LAYERS)
        return net

    def preprocess(self,image):
        return image - self.mean_pixel


    def unprocess(self,image):
        return image + self.mean_pixel
