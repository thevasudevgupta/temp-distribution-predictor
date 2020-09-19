"""MODEL ARCHITECTURE

@author: vasudevgupta
"""
import tensorflow as tf

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
        self.downsample1= Downsample(32, input_shape= (64,64,3))
        self.downsample2= Downsample(64)
        self.downsample3= Downsample(128)
        self.downsample4= Downsample(256)
        self.downsample5= Downsample(512)
        self.flatten= tf.keras.layers.Flatten()
        self.dense= tf.keras.layers.Dense(1, activation= 'sigmoid')
        
    def call(self, inputs):
        conditions, unknown= inputs
        x= tf.concat([conditions, unknown], axis= -1)
        ds1= self.downsample1(x + 1.0*tf.random.uniform(tf.shape(x)))
        ds2= self.downsample2(ds1 + 1.0*tf.random.uniform(tf.shape(ds1)))
        ds3= self.downsample3(ds2 + 1.0*tf.random.uniform(tf.shape(ds2)))
        ds4= self.downsample4(ds3 + 1.0*tf.random.uniform(tf.shape(ds3)))
        ds5= self.downsample5(ds4 + 1.0*tf.random.uniform(tf.shape(ds4)))
        x= self.flatten(ds5)
        x= self.dense(x + 0.05*tf.random.uniform(tf.shape(x)))
        return x
