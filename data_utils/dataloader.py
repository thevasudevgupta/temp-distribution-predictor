# __author__ = 'Vasudev Gupta'

import tensorflow as tf
import numpy as np
from sklearn.utils import shuffle

from data_utils.utils import load_array
BATCH_SIZE = 32


def make_dataset(config, mode="2d", configuration_ls=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]):

    val_size = config['cgan'].get('validation_size', 400)

    if mode == "3d":
        arr = load_array("./dataset/3d/array_format/")
        arr = arr.transpose(0, 2, 3, 1)
        print("SHAPE OF LOADED ARRAY:", arr.shape)
        conditions = arr[:, :, :, :-1]
        real_data= arr[:,:,:,-1:]
        conditions, real_data = shuffle(conditions, real_data, random_state=49)

        conditions = np.pad(conditions, ((0,0), (37,38), (27,28), (0,0)), constant_values=-1)
        real_data = np.pad(real_data, ((0,0), (37,38), (27,28), (0,0)), constant_values=-1)

        conditions = tf.convert_to_tensor(conditions, dtype=tf.float32)
        real_data = tf.convert_to_tensor(real_data, dtype=tf.float32)

        train_dataset = tf.data.Dataset.from_tensor_slices((conditions[:-val_size], real_data[:-val_size])).shuffle(config['cgan'].get('buffer_size', 400)).batch(config['cgan'].get('batch_size', BATCH_SIZE))
        val_conditions = conditions[-val_size:]
        val_real_data = real_data[-val_size:]

        return train_dataset, (val_conditions, val_real_data)

    arr = np.zeros(shape=(1, 51, 51, 3))
    for i in configuration_ls:
        configuration = load_array(f'../dataset/arr_format/config{i}')
        # configuration = call_function(i, configuration)
        arr = np.concatenate([arr, configuration], axis=0)
    arr = arr[1:]

    conditions = arr[:, :, :, :-1]
    real_data= arr[:,:,:,-1:]

    conditions, real_data = shuffle(conditions, real_data, random_state=49)

    conditions = np.pad(conditions, ((0, 0), (7, 6), (7, 6), (0, 0)), constant_values=-1)
    real_data = np.pad(real_data, ((0, 0), (7, 6), (7, 6), (0, 0)), constant_values=-1)

    conditions = tf.convert_to_tensor(conditions, dtype=tf.float32)
    real_data = tf.convert_to_tensor(real_data, dtype=tf.float32)

    train_dataset = tf.data.Dataset.from_tensor_slices((conditions[:-val_size], real_data[:-val_size])).shuffle(config['cgan'].get('buffer_size', 400)).batch(config['cgan'].get('batch_size', BATCH_SIZE))
    val_conditions = conditions[-val_size:]
    val_real_data = real_data[-val_size:]

    return train_dataset, (val_conditions, val_real_data)
