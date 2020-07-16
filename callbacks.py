"""Lets create custom Callbacks

@author: vasudevgupta
"""
import tensorflow as tf

class LearningRate(tf.keras.optimizers.schedules.LearningRateSchedule):
    
    def __init__(self, config):
        super(LearningRate, self).__init__()
        self.base_lr= float(config['learning_rate'].get('base_lr', 1e-4))
        self.max_lr= float(config['learning_rate'].get('max_lr', 1e-2))
        self.step_size= tf.cast(config['learning_rate'].get('step_size', 2000), tf.float32)
        
    def __call__(self, step):
        cycle= tf.floor(1 + step/(2 * self.step_size))
        x= tf.abs(step/self.step_size - 2*cycle + 1)
        return self.base_lr + (self.max_lr - self.base_lr)*tf.maximum(tf.cast(0, tf.float32), 1-x)
    
    def get_config(self):
        return {
            'base_lr': self.base_lr,
            'max_lr': self.max_lr,
            'step_size': self.step_size
            }
