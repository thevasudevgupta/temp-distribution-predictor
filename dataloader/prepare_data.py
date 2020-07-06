#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  9 15:09:59 2020

@author: vasudevgupta
"""
import os
import numpy as np
import tensorflow as tf
import pandas as pd
import re

os.chdir('/Users/vasudevgupta/Desktop/GitHub/Prof_Prabhu_project')

os.chdir('dataset/ansys_format/config1')
file_names= os.listdir()

file_name= file_names[2]
df= pd.read_csv(file_name)
df.columns= ['x','y','z','temp']

xunique= np.arange(0, 101, 2)/100

df['x']= df['x'].apply(lambda x: np.round(x, 2))
df['y']= df['y'].apply(lambda x: np.round(x, 2))

df['x']= df['x'].apply(lambda x: x if (100*x)%2 == 0 else np.nan)
df['y']= df['y'].apply(lambda x: x if (100*x)%2 == 0 else np.nan)

df= df.sort_values('x').dropna().reset_index(drop= True).drop('z', axis=1)







