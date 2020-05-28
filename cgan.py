
"""
cGAN implementation

@author: vasudevgupta
"""

import tensorflow as tf
import numpy as np

def downsample(filters, kernel_size= (3,3), strides= 2, padding= 'same'):
    model= tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters= filters, kernel_size= kernel_size,
                               strides= strides, padding= padding,
                               use_bias= False, kernel_initializer= 'he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()
        ])
    return model

def upsample(filters, kernel_size= (3,3), strides= 2, padding= 'same'):
    model= tf.keras.models.Sequential([
        tf.keras.layers.Conv2DTranspose(filters= filters, kernel_size= kernel_size,
                               strides= strides, padding= padding,
                               use_bias= False, kernel_initializer= 'he_normal'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.LeakyReLU()
        ])
    return model

class Generator(tf.keras.Model):
    """
    GENERATOR CLASS
    """
    
    def __init__(self):
        super(Generator, self).__init__()
        self.downsample1= downsample(32)
        self.downsample2= downsample(64)
        self.downsample3= downsample(128)
        self.downsample4= downsample(256)
        self.downsample5= downsample(512)
        self.downsample6= downsample(1024, kernel_size= (2,2))
        
        self.upsample6= upsample(512, kernel_size= (2,2))
        self.upsample5= upsample(256)
        self.upsample4= upsample(128)
        self.upsample3= upsample(64)
        self.upsample2= upsample(32)
        self.upsample1= upsample(1)
    
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
        self.downsample1= downsample(32)
        self.downsample2= downsample(64)
        self.downsample3= downsample(128)
        self.downsample4= downsample(256)
        self.downsample5= downsample(512)
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
def train_step(conditions, real_data, params):
    with tf.GradientTape() as gtape, tf.GradientTape() as dtape:
        for k in range(params.k):
            fake_data= generator(conditions)
            fake_prob= discriminator(conditions, fake_data)
            real_prob= discriminator(conditions, real_data)
            fake_loss= discriminator_bce(tf.zeros_like(fake_prob), fake_prob)
            real_loss= discriminator_bce(tf.ones_like(real_prob), real_prob)
            discriminator_loss= real_loss + fake_loss
        generator_loss= generator_bce(tf.ones_like(fake_prob), fake_prob)
        
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
    
    def train(self, conditions, real_data, params):
        # self.restore_checkpoint(params)
        for epoch in range(1, 1+params.epochs):
            generator_loss, ggrads, discriminator_loss, dgrads= train_step(conditions, real_data, params)
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

## just take an example
arr= np.zeros(shape= (64,64,1))
arr[:, 0,:]= 70
arr[:, -1,:]= 100
arr[0, :,:]= 10
arr[-1,:,:]= 60
bc= tf.expand_dims(tf.convert_to_tensor(arr, dtype= tf.float32), 0)
real_data= bc
conditions= bc
gan= cGAN()
generator_loss, ggrads, discriminator_loss, dgrads= gan.train(bc, bc, params)
