"""LET'S CONVERT DATASET INTO REQUIRED FORMAT

@author: vasudevgupta
"""
import pandas as pd
import numpy as np
import re

import yaml
import time
import logging
import os

os.chdir('/Users/vasudevgupta/Desktop/GitHub/Prof_Prabhu_project')
from dataloader.dataset_utils import *

logger= logging.getLogger(__name__)

def pipeline(file_names, configuration_name, save_directory):
    for file_no, file_name in enumerate(file_names):
        df= read_file(file_name)
        dim_arr= len(df['x'].unique())
        idx_xy= get_mapping_idx(df)
        # special_idx= row_no + col_no + row_no*(num_cols-1)
        special_idx= idx_xy.apply(lambda x: x[0] + x[1] + (dim_arr-1)*x[0])
        new_idx= pd.concat([idx_xy, special_idx], 1)
        
        conditions_temp_arr, conditions_heatflux_arr= get_conditions_arr(file_name, dim_arr)
        temp_arr= get_temp_arr(special_idx, df['temp'], dim_arr)
        
        conditions_temp_arr= conditions_temp_arr[:,:,np.newaxis]
        conditions_heatflux_arr= conditions_heatflux_arr[:,:,np.newaxis]
        temp_arr= temp_arr[:,:,np.newaxis]
        
        arr_for_saving= np.concatenate([conditions_temp_arr, conditions_heatflux_arr, temp_arr], axis= -1)
        save_array(arr_for_saving, configuration_name + '_eg' + str(file_no+1), directory=save_directory)
    logger.info('1st channel: conditions_temp_arr, 2nd channel: conditions_heatflux_arr, 3rd channel: temp_arr')
    return 

if __name__ == '__main__':
    
    config= yaml.safe_load(open('config.yaml', 'r'))
    config= config.get('dataloader', None)
    # use this code for config7-10
    configuration_name= 'config6'
    arr_dir= 'dataset/arr_format/'
    ansys_dir= 'dataset/ansys_format/'
    main_dir= '/Users/vasudevgupta/Desktop/GitHub/Prof_Prabhu_project/'
    
    os.chdir(main_dir + ansys_dir + configuration_name)
    file_names= os.listdir()
    if '.DS_Store' in file_names: file_names.remove('.DS_Store')
    
    pipeline(file_names, configuration_name, save_directory= main_dir + arr_dir + configuration_name)
    
    data= load_array(directory= main_dir+arr_dir+configuration_name)
    print(data.shape)
