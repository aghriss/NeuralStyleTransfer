# Copyright (c) 2015-2017 Anish Athalye. Released under GPLv3.

import os
import numpy as np
import vgg,utils,config
import tensorflow as tf

# default arguments

class Artist(object):
    
    def __init__(self):
        
        self.graph = tf.Graph()
        self.session = tf.Session()
        
        self.content_image = utils.imread(config.CONTENT_PATH)
        self.style_images = [utils.imread(style) for style in config.STYLE_PATH]
        
        self.target_shape = self.content_image.shape
        self.shape = (1,) + self.target_shape
        self.style_shapes = [(1,) + style.shape for style in self.style_images]
        image = tf.placeholder('float', shape=style_shapes[i])
        target_net = self.vgg_net.net_preloaded(image, config.POOLING)
        
        
        self.vgg_net =  vgg.VGG_NET()
        self.load_features()
        self.load_loss()
        

    def init_weights(self):
        for i in range(len(style_images)):
            style_scale = config.STYLE_SCALE
            self.style_images[i] = utils.imresize(self.style_images[i],
                                     style_scale *self.target_shape[1] / self.style_images[i].shape[1])
    
            self.style_blend_weights = [1.0/len(self.style_images) for _ in self.style_images]
       
        
    
    def save_current_image(self):
        combined_rgb = image
        losses.append(loss)
        np.savetxt("losses7.csv",losses)
        if iteration is not None:
            imsave(output_file,str(iteration), combined_rgb)
        else:
            imsave(output_file,"final", combined_rgb)
            

    def load_features(self):

        layer_weight = 1.0
        style_layers_weights = {}
        for style_layer in config.STYLE_LAYERS:
            style_layers_weights[style_layer] = layer_weight
            layer_weight *= style_layer_weight_exp

        # normalize style layer weights
        layer_weights_sum = 0
        for style_layer in config.STYLE_LAYERS:
            layer_weights_sum += style_layers_weights[style_layer]
            
        for style_layer in config.STYLE_LAYERS:
            style_layers_weights[style_layer] /= layer_weights_sum
        
        g = tf.Graph()
        with g.as_default(), tf.Session() as sess:
            image = tf.placeholder('float', shape=self.shape)
            target_net = self.vgg_net.net_preloaded(image, config.POOLING)
            content_pre = np.array([self.vgg_net.preprocess(self.content)])
        
        for layer in config.CONTENT_LAYERS:
            self.content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # compute style features in feedforward mode
        for i in range(len(self.styles)):
            with self.graph.as_default(), tf.session as sess:
                image = tf.placeholder('float', shape=style_shapes[i])
                net = vgg_net.net_preloaded(image, pooling)
                style_pre = np.array([vgg_net.preprocess(styles[i])])
                for layer in STYLE_LAYERS:
                    features = net[layer].eval(feed_dict={image: style_pre})
                    features = np.reshape(features, (-1, features.shape[3]))
                    gram = np.matmul(features.T, features) / features.size
                    style_features[i][layer] = gram

    def load_loss(self):
        # Total loss
        self.calculate_content_loss()
        self.calcualte_style_loss()
        self.TV_loss = utils.TV_loss(self.solution)
        
        self.loss = (config.style_weight*self.style_loss()
                    +config.content_weight*self.content_loss
                    +config.TV_weight*self.TV_loss)


    def calculate_content_loss(self,image):
        # Content loss
        content_layers_weights = {}
        content_layers_weights['relu4_2'] = content_weight_blend
        content_layers_weights['relu5_2'] = 1.0 - content_weight_blend

        self.content_loss = 0
        
        for content_layer in CONTENT_LAYERS:
            self.content_loss += (content_layers_weights[content_layer] *
                                tf.reduce_mean(
                                tf.square(net[content_layer] - content_features[content_layer]) ))
        
    def calculate_style_loss(self,image):
        # style loss
        self.style_loss = 0
        for i in range(len(styles)):
            style_losses = []
            for style_layer in config.STYLE_LAYERS:
                layer = net[style_layer]
                _, height, width, number = map(lambda i: i.value, layer.get_shape())
                size = height * width * number
                feats = tf.reshape(layer, (-1, number))
                gram = tf.matmul(tf.transpose(feats), feats) / size
                style_gram = style_features[i][style_layer]
                style_loss += style_layers_weights[style_layer]*tf.reduce_mean(tf.square(gram - style_gram))
            self.style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)
