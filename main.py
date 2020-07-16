"""cGAN MAIN FILE 

@author: vasudevgupta
"""
import tensorflow as tf

import yaml
import argparse
import os
import wandb
import logging

from model.cgan import *
from model.architecture import Generator, Discriminator
from dataloader.tf_dataset import make_dataset

if __name__ == '__main__':
    os.chdir('/Users/vasudevgupta/Desktop/GitHub/Prof_Prabhu_project')
    logger= logging.getLogger(__name__)
    
    logger.setLevel(logging.INFO)
    file_handler= logging.FileHandler('logs/results.log')
    formt= logging.Formatter('%(levelname)s:%(name)s:%(message)s')
    file_handler.setFormatter(formt)
    logger.addHandler(file_handler)
    
    parser= argparse.ArgumentParser()
    
    parser.add_argument('--config', type= str, default= 'config.yaml', help= 'filename for configuration')
    parser.add_argument('--model_name', type= str, default= 'cgan_model', help= 'model no')
    parser.add_argument('--save_model', type= bool, default= False, help= 'whether to save model')
    parser.add_argument('--restore_model', type= bool, default= False, help= 'whether to restore saved weights')
    
    args= parser.parse_args()
    
    config= yaml.safe_load(open(args.config, 'r'))
    logger.info(config)
    
    train_dataset, val_dataset= make_dataset(config)
    val_conditions, val_real_data= val_dataset
    
    wandb.init(project= 'temp_distribution_prediction', config= config, name= args.model_name)
    config= wandb.config
    
    generator= Generator()
    discriminator= Discriminator()
    
    model= cGAN(generator= generator, discriminator= discriminator)
    model.icompile(config)
    
    final_metric, grads= model.train(train_dataset, val_conditions, val_real_data, config,
                         restore_model= args.restore_model, save_model= args.save_model)
    
    logger.info(final_metric)
    model.save_checkpoints()
    
    
    
    
    
    