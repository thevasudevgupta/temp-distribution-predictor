"""COMBINE GENEARTOR AND DISCRIMINATOR

@author: vasudevgupta
"""
import numpy as np
import tensorflow as tf

from modeling_utils.callbacks import LearningRate

from tqdm import tqdm
import wandb
import logging
import os
import time

logger= logging.getLogger(__name__)

logger.setLevel(logging.INFO)
file_handler= logging.FileHandler('./logs/train.log')
form= logging.Formatter('%(levelname)s:%(name)s:%(message)s')
file_handler.setFormatter(form)
logger.addHandler(file_handler)

class cGAN(object):
    
    def __init__(self, generator, discriminator):
        self.generator= generator
        self.discriminator= discriminator
    
    def icompile(self, config):
        self.config= config
        
        self.discriminator_bce= tf.keras.losses.BinaryCrossentropy(from_logits= False, label_smoothing=0.1)
        
        self.generator_bce= tf.keras.losses.BinaryCrossentropy(from_logits= False, label_smoothing=0.1)
        self.generator_mse= tf.keras.losses.MeanSquaredError()
        
        self.glearning_rate= LearningRate(self.config['generator'])
        self.dlearning_rate= LearningRate(self.config['discriminator'])
        
        if config['generator'].get('optimizer') == 'adam':
            self.goptimizer= tf.keras.optimizers.Adam(self.glearning_rate)
        else:
            self.goptimizer= tf.keras.optimizers.SGD(self.glearning_rate)
        
        if config['discriminator'].get('optimizer') == 'sgd':
            self.doptimizer= tf.keras.optimizers.SGD(self.dlearning_rate)
        else: 
            self.doptimizer= tf.keras.optimizers.Adam(self.dlearning_rate)
    
        self.mse_metric= tf.keras.metrics.MeanSquaredError()
        self.mae_metric= tf.keras.metrics.MeanAbsoluteError()
        
    def train(self, train_dataset, val_conditions, val_real_data, config, restore_model= True, save_model= False):
        if restore_model: self.restore_checkpoint(self.config['cgan'].get('ckpt_dir', 'cgan_ckpt'))
        start= time.time()
        for epoch in range(1, 1+config['cgan'].get('epochs', 2)):
            
            gloss_ls= []
            dloss_ls= []
            tr_generator_mse_ls= []
            
            for conditions, real_data in tqdm(train_dataset, desc=f"running epoch-{epoch}"):
                
                history= self.train_step(conditions, real_data, val_conditions, val_real_data)
                
                gloss_ls.append(history['metrics'].get('gloss', 0))
                dloss_ls.append(history['metrics'].get('dloss', 0))
                tr_generator_mse_ls.append(history['metrics'].get('tr_generator_mse', 0))
            
            final_metric= {
                'gloss': tf.reduce_mean(gloss_ls).numpy(),
                'dloss': tf.reduce_mean(dloss_ls).numpy(),
                'tr_generator_mse': tf.reduce_mean(tr_generator_mse_ls).numpy(),
                'val_generator_mae': history['metrics']['val_generator_mae'].numpy(),
                'val_generator_mse': history['metrics']['val_generator_mse'].numpy()
                }
            wandb.log(final_metric)
            
            self.mse_metric.reset_states()
            self.mae_metric.reset_states()
        
            logger.info(f"epoch-{epoch} done -------- metrics_output- {final_metric}")
            print((f"epoch-{epoch} done -------- metrics_output- {final_metric}"))
            
            if epoch%5 == 0:
                if save_model: self.save_checkpoints()
        
        logger.info(f'TOTAL TIME TAKEN: {time.time()-start}')
        
        final_grads= {}
        final_grads['ggrads'] = history['ggrads']
        final_grads['dgrads'] = history['dgrads']
        
        return final_metric, final_grads
        
    @tf.function
    def train_step(self, conditions, real_data, val_conditions, val_real_data):

        with tf.GradientTape() as gtape, tf.GradientTape() as dtape:

            for _ in range(self.config['cgan'].get('k', 1)):

                fake_data= self.generator(conditions)
                
                fake_prob= self.discriminator([conditions, fake_data])
                real_prob= self.discriminator([conditions, real_data])
                # # adding random noise; may help in stablizing
                fake_label= tf.zeros_like(fake_prob) # + self.config['discriminator'].get(epsilon, 0.0)*tf.random.uniform(tf.shape(fake_prob)))
                real_label= tf.ones_like(real_prob) # + self.config['discriminator'].get(epsilon, 0.0)*tf.random.uniform(tf.shape(real_prob)))         
    
                fake_loss= self.discriminator_bce(fake_label, fake_prob)
                real_loss= self.discriminator_bce(real_label, real_prob)
                
                discriminator_loss= real_loss + fake_loss
            
            extra_loss= self.generator_mse(real_data, fake_data)
            generator_loss= self.generator_bce(tf.ones_like(fake_prob), fake_prob) + self.config['generator'].get('lambd', 0)*tf.math.log(extra_loss)

        dgrads= dtape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.doptimizer.apply_gradients(zip(dgrads, self.discriminator.trainable_variables))

        ggrads= gtape.gradient(generator_loss, self.generator.trainable_variables)
        self.goptimizer.apply_gradients(zip(ggrads, self.generator.trainable_variables))

        self.evaluate(val_conditions, val_real_data)

        metrics= {
            'gloss': generator_loss,
            'dloss': discriminator_loss,
            'tr_generator_mse': extra_loss,
            'val_generator_mae': self.mae_metric.result(),
            'val_generator_mse': self.mse_metric.result()
            }

        return {'metrics': metrics, 'ggrads': ggrads, 'dgrads': dgrads}

    def evaluate(self, val_conditions, val_real_data):
        val_fake_pred= self.generator(val_conditions)
        self.mse_metric.update_state(val_real_data, val_fake_pred)
        self.mae_metric.update_state(val_real_data, val_fake_pred)
        return {'mae': self.mae_metric.result(), 'mse': self.mse_metric.result()}

    def save_checkpoints(self):
        checkpoint_dir = self.config['cgan'].get('ckpt_dir', 'cgan_ckpt')
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        ckpt= tf.train.Checkpoint(generator_optimizer= self.goptimizer,
                                  discriminator_optimizer= self.doptimizer,
                                  generator= self.generator,
                                  discriminator= self.discriminator)
        ckpt.save(file_prefix= checkpoint_prefix)
            
    def restore_checkpoint(self, ckpt_dir):
        checkpoint_dir = ckpt_dir
        ckpt= tf.train.Checkpoint(generator_optimizer= self.goptimizer,
                                  discriminator_optimizer= self.doptimizer,
                                  generator= self.generator,
                                  discriminator= self.discriminator)
        ckpt.restore(tf.train.latest_checkpoint(checkpoint_dir))
        
    def get_prediction(self, conditions, ckpt_dir):
        self.restore_checkpoint(ckpt_dir)
        prediction= self.generator(conditions)
        return prediction
