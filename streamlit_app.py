# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:10:29 2022

@author: Javier
"""

"""
# Predicción de incumplimiento crediticio 
"""

import pandas as pd
import streamlit as st

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
"""
## Conjuno de entrenamiento
"""
st.write(train.head())

"""
## Conjunto de datos prueba
"""
st.write(test.head())



