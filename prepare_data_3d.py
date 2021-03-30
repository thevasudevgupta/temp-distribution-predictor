# __author__ = 'Vasudev Gupta'

# idx_Thout_Vhout_Tcin_Vcin
# Tcout = 0.8 x Thout
# Vcout = 0.8 x Vhout

# x = 50 x cos(theta)
# y = 50 x sin(theta)

# Matrix -> theta (181), height (201)

"""
USAGE:
    Run this script to prepare convert data from .csv format to 2D matrix format

COMMAND:
    `python prepare_data_3d.py`
"""

import numpy as np
import pandas as pd

from tqdm import tqdm
import logging
import os

from data_utils.utils import load_array, save_array

logger = logging.getLogger(__name__)

ANSYS_DATA_DIR = "./dataset/3d/ansys_format/"
ARRAY_DATA_DIR = "./dataset/3d/array_format/"


def pipeline(load_dir, save_dir=None):

    file_names = os.listdir(load_dir)
    if '.DS_Store' in file_names: file_names.remove('.DS_Store')

    for file_name in tqdm(file_names):

        # load dataset
        df = pd.read_table(os.path.join(load_dir, file_name))
        assert df.iloc[7, 0] == "% x,y,z,T (K)", f"{file_name} not in expected format"
        assert df.shape == (36389, 1), f"{file_name} not in expected format"
        df = df.iloc[8:, 0]
        df = df.str.split(',', expand=True)
        df.columns = ['x', 'y', 'z', 'T']
        for col in df.columns:
            df[col] = df[col].apply(lambda x: round(float(x), 4))
        assert df.shape == (181*201, 4), f"Issues in {file_name}"

        # preprocess
        ls = []
        for _ in range(df.shape[0]//181):
            ls.extend(list(range(181)))
        df["theta"] = ls
        # z, theta, T are of interest

        # temperature array for theta vs height
        temp_arr = np.zeros(shape=(201*181, 1))
        for idx, val in zip(range(df.shape[0]), df["T"]):
            np.put(temp_arr, idx, val)
        temp_arr = temp_arr.reshape(201, 181)
        temp_arr = temp_arr.transpose(1, 0)
        # -> (181, 201)

        # get boundary conditions
        _, tho, vho, tci, vci = file_name.split("_")
        tho = float(tho)
        vho = float(vho)
        tci = float(tci)
        vci = float(vci[:-3])
        tco = 0.8 * (tho - 273.17) + 273.17
        vco = 0.8 * vho

        to_channel = np.zeros_like(temp_arr)        
        to_channel[:90, :] = tco
        to_channel[90:, :] = tho

        vo_channel = np.zeros_like(temp_arr)
        vo_channel[:90, :] = vco
        vo_channel[90:, :] = vho

        ti_channel = np.zeros_like(temp_arr)
        ti_channel[:, : 201//4] = tci
        ti_channel[:, 201//4: 2*(201//4)] = tci + 25
        ti_channel[:, 2*(201//4): 3*(201//4)] = tci + 50
        ti_channel[:, 3*(201//4):] = tci + 100

        vi_channel = np.zeros_like(temp_arr)
        vi_channel.fill(vci)

        # Save in numpy format
        arr_for_saving = np.stack([to_channel, vo_channel, ti_channel, vi_channel, temp_arr])
        if save_dir is not None:
            save_array(arr_for_saving, file_name[:-4], directory=save_dir)

    print("Preprocessed data saved in ", ARRAY_DATA_DIR)
    logger.info('1st channel: TEMPERATURE OUTSIDE, 2nd channel: VELOCITY OUTSIDE, 3rd channel: TEMPERATURE INSIDE, \
        4th channel: VELOCITY INSIDE, 5th channel: TEMPERATURE PREDICTION')


if __name__ == '__main__':

    pipeline(ANSYS_DATA_DIR, save_dir=ARRAY_DATA_DIR)

    data = load_array(directory=ARRAY_DATA_DIR)
    print('shape of data is', data.shape)
