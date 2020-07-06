"""SOME IMPORTANT FUNCTION/ CLASSES

@author: vasudevgupta
"""
import re
import pandas as pd
import numpy as np

import os
import logging
logger= logging.getLogger(__name__)

# FILE NAME FORMAT
# A-Left edge
# B-Bottom edge
# C- Right edge
# D-top edge
# (Temperature Value, Heat flux)

def get_conditions_arr(file_name,  dim_arr, return_config= False):
    configuration_name= re.findall(r'[Cc]onfig[0-9]+', file_name)[0]
    configuration= {'configuration_name': configuration_name}
    # A= re.findall(r'[Aa][(_]-?\d+,-?\d+[)_]', file_name)[0]
    for name in list('ABCD'):
        cf= re.findall(r'[{}][(_]-?\d+,-?\d+[)_]'.format(name), file_name)[0]
        configuration[f'{name}_temp'], configuration[f'{name}_hf']= re.findall(r'[-?\d]+', cf)

    conditions_temp_arr= np.zeros(shape= (dim_arr, dim_arr))
    conditions_heatflux_arr= np.zeros(shape= (dim_arr, dim_arr))
    
    conditions_temp_arr[:, 0]= configuration['A_temp']
    conditions_temp_arr[-1, :]= configuration['B_temp']
    conditions_temp_arr[:, -1]= configuration['C_temp']
    conditions_temp_arr[0, :]= configuration['D_temp']
    
    # handling corner points
    conditions_temp_arr[0,0]= max(conditions_temp_arr[0,1], conditions_temp_arr[1,0]) if (conditions_temp_arr[0,1] == 0 or conditions_temp_arr[1,0] == 0) else np.mean([conditions_temp_arr[0,1], conditions_temp_arr[1,0]])
    conditions_temp_arr[0,-1]= max(conditions_temp_arr[0,-2], conditions_temp_arr[1,-1]) if (conditions_temp_arr[0,-2] == 0 or conditions_temp_arr[1,-1] == 0) else np.mean([conditions_temp_arr[0,-2], conditions_temp_arr[1,-1]])
    conditions_temp_arr[-1,0]= max(conditions_temp_arr[-2,0], conditions_temp_arr[-1,1]) if (conditions_temp_arr[-2,0] == 0 or conditions_temp_arr[-1,1] == 0) else np.mean([conditions_temp_arr[-1,1], conditions_temp_arr[-2,0]])
    conditions_temp_arr[-1,-1]= max(conditions_temp_arr[-2,-1], conditions_temp_arr[-1,-2]) if (conditions_temp_arr[-2,-1] == 0 or conditions_temp_arr[-1,-2] == 0) else np.mean([conditions_temp_arr[-2,-1], conditions_temp_arr[-1,-2]])
    
    conditions_heatflux_arr[:, 0]= configuration['A_hf']
    conditions_heatflux_arr[-1, :]= configuration['B_hf']
    conditions_heatflux_arr[:, -1]= configuration['C_hf']
    conditions_heatflux_arr[0, :]= configuration['D_hf']
    
    # handling corner points
    conditions_heatflux_arr[0,0]= min(conditions_heatflux_arr[0,1], conditions_heatflux_arr[1,0]) if (conditions_heatflux_arr[0,1] == 0 or conditions_heatflux_arr[1,0] == 0) else np.mean([conditions_heatflux_arr[0,1], conditions_heatflux_arr[1,0]])
    conditions_heatflux_arr[0,-1]= min(conditions_heatflux_arr[0,-2], conditions_heatflux_arr[1,-1]) if (conditions_heatflux_arr[0,-2] == 0 or conditions_heatflux_arr[1,-1] == 0) else np.mean([conditions_heatflux_arr[0,-2], conditions_heatflux_arr[1,-1]])
    conditions_heatflux_arr[-1,0]= min(conditions_heatflux_arr[-2,0], conditions_heatflux_arr[-1,1]) if (conditions_heatflux_arr[-1,1] == 0 or conditions_heatflux_arr[-2,0] == 0) else np.mean([conditions_heatflux_arr[-1,1], conditions_heatflux_arr[-2,0]])
    conditions_heatflux_arr[-1,-1]= min(conditions_heatflux_arr[-2,-1], conditions_heatflux_arr[-1,-2]) if (conditions_heatflux_arr[-2,-1] == 0 or conditions_heatflux_arr[-1,-2] == 0) else np.mean([conditions_heatflux_arr[-2,-1], conditions_heatflux_arr[-1,-2]])
    
    if return_config: return conditions_temp_arr, conditions_heatflux_arr, configuration
    return conditions_temp_arr, conditions_heatflux_arr

# with open(file_name) as file:
#     data= file.read()

# idx= re.search(r'%\s?[xX]', data).span()[-1] - 1
# data= data[idx:].split('\n')
# if '' in data: data.remove('')

def read_file(file_name, check_assertion= True):
    table= pd.read_table(file_name)
    df_col= table.iloc[:,0].apply(lambda x: str(x).split(','))
    df_col= df_col.apply(lambda x: x if len(x) == 3 else np.nan)
    df_col.dropna(inplace= True)
    df_col= df_col[2:]
    
    df= pd.DataFrame()
    df['x']= df_col.apply(lambda x: np.round(float(x[0]), 2))
    df['y']= df_col.apply(lambda x: np.round(float(x[1]), 2))
    df['temp']= df_col.apply(lambda x: float(x[2]))
    df= df.reset_index(drop=True)
    
    if check_assertion:
        assert(df.shape == (2601,3))
        assert(df.iloc[0,0] == float(0) and df.iloc[0,1] == float(0))
        assert(df.iloc[-1,0] == float(1) and df.iloc[-1,1] == float(1))
    return df

def get_mapping_idx(df):
    num_idx= list(range(df.shape[0]))
    str_idx= [str(n) for n in num_idx]
    
    assert(all(df['x'].unique() == df['y'].unique()))
    map_idx= dict(zip(df['x'].unique(), str_idx))
    
    ds= pd.DataFrame()
    ds['map_x']= df['x'].map(map_idx)
    ds['map_y']= df['y'].map(map_idx)
    # x = row in figure; means its corresponds values in diff cols
    ds['idx_xy']= ds['map_y'].str.cat(ds['map_x'], sep= ',')
    ds['idx_xy']= ds['idx_xy'].apply(lambda x: [int(idx) for idx in x.split(',')])
    return ds['idx_xy']

def get_temp_arr(special_idx, df_col, dim_arr):
    temp_arr= np.zeros(shape= (dim_arr*dim_arr, 1))
    # for row in df['map_y']:
    #     for col in df['map_x']:
    #         new_df= df[df['map_x'] == col]
    #         temp_arr[row, col]= new_df[new_df['map_y'] == row]['temp'].values[0]
    for idx, val in zip(special_idx, df_col):
        np.put(temp_arr, idx, val)
    return temp_arr.reshape(dim_arr, dim_arr)

def save_array(array, file_no, directory):
    with open(directory + f"/{file_no}.npy", "wb") as file:
        np.save(file, array)
    return 

def load_array(directory):
    arr= []
    list_dir= os.listdir(directory)
    if '.DS_Store' in list_dir: list_dir.remove('.DS_Store')
    for file in list_dir:
        arr.append(np.load(directory+'/'+file))
    return np.array(arr)
