#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('matplotlib', 'inline')
import sys
import numpy as np
np.set_printoptions(threshold = sys.maxsize)

import matplotlib.pyplot as plt
import seaborn as sns


def c1_modify(config_arr):
    c1_bin = np.zeros((51,51))
    # Temperature Boundary Condition
    c1_bin[:, 0] = 1
    # Temperature Sensor Measurements
    c1_bin[5, 15:50:10] = 1
    c1_bin[15, 25:50:10] = 1
    c1_bin[25, 35:50:10] = 1
    c1_bin[35, 45] = 1
    ans = np.multiply(c1_bin,config_arr[:,:,:,2])
    config_arr[:,:,:,2] = np.where(ans != 0,ans,-1)
    return config_arr

def c2_modify(config_arr):
    c2_bin = np.zeros((51,51))
    # Temperature Boundary Condition
    c2_bin[:, 0] = 1
    # Temperature Sensor Measurements
    c2_bin[10::15, 45] = 1
    c2_bin[10::15, 35] = 1
    c2_bin[10::15, 25] = 1
    c2_bin[25, 15] = 1
    ans = np.multiply(c2_bin,config_arr[:,:,:,2])
    config_arr[:,:,:,2] = np.where(ans != 0,ans,-1)
    return config_arr

def c3_modify(config_arr):
    c3_bin = np.zeros((51, 51))
    # Temperature Boundary Condition
    c3_bin[:, 0] = 1
    c3_bin[-1, :] = 1
    # Temperature Sensor Measurements
    c3_bin[5, 15:50:10] = 1
    c3_bin[15, 25:50:10] = 1
    c3_bin[25, 35:50:10] = 1
    c3_bin[35, 45] = 1
    ans = np.multiply(c3_bin,config_arr[:,:,:,2])
    config_arr[:,:,:,2] = np.where(ans != 0,ans,-1)
    return config_arr

def c4_modify(config_arr):
    c4_bin = np.zeros((51,51))
    # Temperature Boundary Condition
    c4_bin[:, [0, -1]] = 1
    # Temperature Sensor Measurements
    c4_bin[2, 9:42:8] = 1
    c4_bin[10, 13:38:12] = 1
    c4_bin[20, [20, 30]] = 1
    ans = np.multiply(c4_bin,config_arr[:,:,:,2])
    config_arr[:,:,:,2] = np.where(ans != 0,ans,-1)
    return config_arr

# In[43]:

def c5_modify(config_arr):
    c5_bin = np.zeros((51, 51))
    # Temperature Boundary Condition
    c5_bin[:, 0] = 1
    # Temperature Sensor Measurements
    c5_bin[10:21:10, 25::10] = 1
    c5_bin[30:50:10, 30:50:10] = 1
    ans = np.multiply(c5_bin,config_arr[:,:,:,2])
    config_arr[:,:,:,2] = np.where(ans != 0,ans,-1)
    return config_arr

# In[44]:

def c6_modify(config_arr):
    c6_bin = np.zeros((51, 51))
    # Temperature Boundary  Condition
    c6_bin[:, 0] = 1
    # Temperature Sensor Measurements
    c6_bin[10:50:10, 45] = 1
    c6_bin[15:45:10, 35] = 1
    c6_bin[[20, 30], 25] = 1
    c6_bin[25, 15] = 1
    ans = np.multiply(c6_bin,config_arr[:,:,:,2])
    config_arr[:,:,:,2] = np.where(ans != 0,ans,-1)
    return config_arr

# In[45]:

def c7_modify(config_arr):
    c7_bin = np.zeros((51, 51))
    # Temperature Boundary Condtion
    c7_bin[:,[0, -1]] = 1
    # Temperature Sensor Measurements
    c7_bin[2, 15:45:10] = 1
    c7_bin[12, [20, 30]] = 1
    c7_bin[38, [20, 30]] = 1
    c7_bin[48, 15:45:10] = 1
    ans = np.multiply(c7_bin,config_arr[:,:,:,2])
    config_arr[:,:,:,2] = np.where(ans != 0,ans,-1)
    return config_arr
    # In[46]:

def c8_modify(config_arr):
    c8_bin = np.zeros((51, 51))
    # Temperature Boundary Condition
    c8_bin[:, 0] = 1
    c8_bin[-1, :] = 1
    # Temperature Sensor Measurements
    c8_bin[5, 15:50:10] = 1
    c8_bin[15, 25:50:10] = 1
    c8_bin[25, 35:50:10] = 1
    c8_bin[35, 45] = 1
    ans = np.multiply(c8_bin,config_arr[:,:,:,2])
    config_arr[:,:,:,2] = np.where(ans != 0,ans,-1)
    return config_arr

# In[47]:

def c9_modify(config_arr):
    c9_bin = np.zeros((51, 51))
    # Temperature Boundary Condition
    c9_bin[:, [0, -1]] = 1
    c9_bin[-1, :] = 1
    # Temperature Sensor Measurements
    c9_bin[2, 10:50:10] = 1
    c9_bin[10, 15:45:10] = 1
    c9_bin[18, [20, 30]] = 1
    c9_bin[25, 25] = 1
    ans = np.multiply(c9_bin,config_arr[:,:,:,2])
    config_arr[:,:,:,2] = np.where(ans != 0,ans,-1)
    return config_arr

# In[48]:

def c10_modify(config_arr):
    c10_bin = np.zeros((51, 51))
    # Temperature Boundary Condition
    c10_bin[:, 0] = 1
    # Temperature Sensor Measurments
    c10_bin[5::20, 45] = 1
    c10_bin[15:45:10, 35] = 1
    c10_bin[[17, 33], 25] = 1
    c10_bin[[20, 30], 15] = 1
    ans = np.multiply(c10_bin,config_arr[:,:,:,2])
    config_arr[:,:,:,2] = np.where(ans != 0,ans,-1)
    return config_arr

def call_function(i,configuration):
    thismodule = sys.modules[__name__]
    fn_name = getattr(thismodule, "c{}_modify".format(i))
    configuration = fn_name(configuration)
    return configuration

