# __author__ = 'Vasudev Gupta'

"""
USAGE:
    Run this script to prepare convert data from .csv format to 2D matrix format

COMMAND:
    `python prepare_data_2d.py --config_name {config_name}`
"""

import pandas as pd
import numpy as np

import logging
import os

from pathlib import Path
import argparse

from data_utils.utils import *

logger= logging.getLogger(__name__)

def pipeline(file_names, configuration_name, save_directory):
  
    for file_no, file_name in enumerate(file_names, start=1):
  
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
        save_array(arr_for_saving, f"{configuration_name}_eg{file_no}", directory=save_directory)

    logger.info('1st channel: conditions_temp_arr, 2nd channel: conditions_heatflux_arr, 3rd channel: temp_arr')
    
    return 

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--config_name', type=str, help='name of config')
    # eg: config1
    parser.add_argument('--load_path', type=str, default='dataset/ansys_format', help='Where to pick data for conversion')
    parser.add_argument('--save_path', type=str, default='dataset/matrix_format', help='Where to put data after conversion')
    args = parser.parse_args()

    file_names = os.listdir(Path(args.load_path, args.config_name))
    if '.DS_Store' in file_names: file_names.remove('.DS_Store')

    pipeline(file_names, args.config_name, save_directory=Path(args.save_path, args.config_name))
    
    data = load_array(directory= args.load_path)
    print('shape of data is', data.shape)
