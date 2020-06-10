
"""
cGAN implementation

@author: vasudevgupta
"""

import tensorflow as tf
import numpy as np
import os

class Downsample(tf.keras.layers.Layer):
    
    def __init__(self, filters, kernel_size= (3,3), strides= 2, padding= 'same', input_shape= None):
        super(Downsample, self).__init__()
        if input_shape == None:
            self.conv= tf.keras.layers.Conv2D(filters= filters, kernel_size= kernel_size,
                                              strides= strides, padding= padding,
                                              use_bias= False, kernel_initializer= 'he_normal')
        else:
            self.conv= tf.keras.layers.Conv2D(filters= filters, kernel_size= kernel_size,
                                              strides= strides, padding= padding, input_shape= input_shape,
                                              use_bias= False, kernel_initializer= 'he_normal')
        self.bn= tf.keras.layers.BatchNormalization()
        self.actn= tf.keras.layers.LeakyReLU()
    
    def call(self, x):
        x= self.conv(x)
        x= self.bn(x)
        x= self.actn(x)
        return x

class Upsample(tf.keras.layers.Layer):
    
    def __init__(self, filters, kernel_size= (3,3), strides= 2, padding= 'same', input_shape= None):
        super(Upsample, self).__init__()
        if input_shape == None:
            self.convt= tf.keras.layers.Conv2DTranspose(filters= filters, kernel_size= kernel_size,
                                              strides= strides, padding= padding,
                                              use_bias= False, kernel_initializer= 'he_normal')
        else:
            self.convt= tf.keras.layers.Conv2DTranspose(filters= filters, kernel_size= kernel_size,
                                              strides= strides, padding= padding, input_shape= input_shape,
                                              use_bias= False, kernel_initializer= 'he_normal')
        self.bn= tf.keras.layers.BatchNormalization()
        self.actn= tf.keras.layers.LeakyReLU()
            
    def call(self, x):
        x= self.convt(x)
        x= self.bn(x)
        x= self.actn(x)
        return x                  
            
class Generator(tf.keras.Model):
    """
    GENERATOR CLASS
    """
    
    def __init__(self):
        super(Generator, self).__init__()
        self.downsample1= Downsample(32, input_shape= (64,64,2))
        self.downsample2= Downsample(64)
        self.downsample3= Downsample(128)
        self.downsample4= Downsample(256)
        self.downsample5= Downsample(512)
        self.downsample6= Downsample(1024, kernel_size= (2,2))
        
        self.upsample6= Upsample(512, kernel_size= (2,2))
        self.upsample5= Upsample(256)
        self.upsample4= Upsample(128)
        self.upsample3= Upsample(64)
        self.upsample2= Upsample(32)
        self.upsample1= Upsample(1)
    
    def call(self, x):
        ds1= self.downsample1(x)
        ds2= self.downsample2(ds1)
        ds3= self.downsample3(ds2)
        ds4= self.downsample4(ds3)
        ds5= self.downsample5(ds4)
        x= self.downsample6(ds5)
        
        up6= self.upsample6(x)
        up5= self.upsample5(tf.concat([up6, ds5], axis= -1))
        up4= self.upsample4(tf.concat([up5, ds4], axis= -1))
        up3= self.upsample3(tf.concat([up4, ds3], axis= -1))
        up2= self.upsample2(tf.concat([up3, ds2], axis= -1))
        up1= self.upsample1(tf.concat([up2, ds1], axis= -1))
        return up1

class Discriminator(tf.keras.Model):
    
    def __init__(self):
        super(Discriminator, self).__init__()
        self.downsample1= Downsample(32, input_shape= (64,64,1))
        self.downsample2= Downsample(64)
        self.downsample3= Downsample(128)
        self.downsample4= Downsample(256)
        self.downsample5= Downsample(512)
        self.flatten= tf.keras.layers.Flatten()
        self.dense= tf.keras.layers.Dense(1, activation= 'sigmoid')
        
    def call(self, conditions, unknown):
        x= tf.concat([conditions, unknown], axis= -1)
        ds1= self.downsample1(x)
        ds2= self.downsample2(ds1)
        ds3= self.downsample3(ds2)
        ds4= self.downsample4(ds3)
        ds5= self.downsample5(ds4)
        x= self.flatten(ds5)
        x= self.dense(x)
        return x

@tf.function    
def train_step(conditions, real_data, params, generator, discriminator, discriminator_bce, generator_bce, generator_mse):
    with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
        for k in range(params.k):
            fake_data= generator(conditions)
            fake_prob= discriminator(conditions, fake_data)
            real_prob= discriminator(conditions, real_data)
            fake_loss= discriminator_bce(tf.zeros_like(fake_prob), fake_prob)
            real_loss= discriminator_bce(tf.ones_like(real_prob), real_prob)
            discriminator_loss= real_loss + fake_loss
        generator_loss= generator_bce(tf.ones_like(fake_prob), fake_prob) + params.lambd*generator_mse(real_data, fake_data)
        
    dgrads= dtape.gradient(discriminator_loss, discriminator.trainable_variables)
    params.doptimizer.apply_gradients(zip(dgrads, discriminator.trainable_variables))
    
    ggrads= gtape.gradient(generator_loss, generator.trainable_variables)
    params.goptimizer.apply_gradients(zip(ggrads, generator.trainable_variables))
    return generator_loss, ggrads, discriminator_loss, dgrads

class cGAN:
    
    def __init__(self):
        self.generator= Generator()
        self.discriminator= Discriminator()
        self.discriminator_bce= tf.keras.losses.BinaryCrossentropy(from_logits= False)
        self.generator_bce= tf.keras.losses.BinaryCrossentropy(from_logits= False)
        self.generator_mse= tf.keras.losses.MeanSquaredError()
    
    def train(self, conditions, real_data, params):
        # self.restore_checkpoint(params)
        for epoch in range(1, 1+params.epochs):
            generator_loss, ggrads, discriminator_loss, dgrads= train_step(conditions, real_data, 
                                                                           params, self.generator, 
                                                                           self.discriminator, 
                                                                           discriminator_bce= self.discriminator_bce, 
                                                                           generator_bce= self.generator_bce,
                                                                           generator_mse= self.generator_mse)
            if epoch%5 == 0:
                self.save_checkpoints(params)
        return generator_loss, ggrads, discriminator_loss, dgrads
        
    def save_checkpoints(self, params):
        checkpoint_dir = 'cgan_ckpt'
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        ckpt= tf.train.Checkpoint(generator_optimizer= params.goptimizer,
                                  discriminator_optimizer= params.doptimizer,
                                  generator= self.generator,
                                  discriminator= self.discriminator)
        ckpt.save(file_prefix= checkpoint_prefix)
        
    def restore_checkpoint(self, params):
        checkpoint_dir = 'cgan_ckpt'
        ckpt= tf.train.Checkpoint(generator_optimizer= params.goptimizer,
                                  discriminator_optimizer= params.doptimizer,
                                  generator= self.generator,
                                  discriminator= self.discriminator)
        ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))    
    
class params:
    pass

params.learning_rate= 0.001
params.goptimizer= tf.keras.optimizers.Adam(params.learning_rate)
params.doptimizer= tf.keras.optimizers.Adam(params.learning_rate)
params.k= 2
params.epochs= 2
params.lambd= 0.01

# x= tf.ones((1,64,64,2))
# y= tf.ones((1,64,64,1))
cgan= cGAN()
generator_loss, ggrads, discriminator_loss, dgrads= cgan.train(x,y, params)
cgan.save_checkpoints(params)