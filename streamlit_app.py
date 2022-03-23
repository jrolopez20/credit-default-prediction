# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 21:10:29 2022

@author: Javier
"""

"""
# Predicción de incumplimiento crediticio 
"""


import streamlit as st
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2

train = pd.read_csv('./input/train.csv')
test = pd.read_csv('./input/test.csv')
"""
### Conjunto de datos de entrenamiento
"""
st.write(train.head())

"""
### Conjunto de datos de prueba
"""
st.write(test.head())

# Strip column names of all spaces and add underscore wherever required
train.columns = ['_'.join(col.split(' ')).lower() for col in train.columns]
test.columns = ['_'.join(col.split(' ')).lower() for col in test.columns]

# Transform to numeric values for years in current job column
years_in_current_job_map = {
    '10+ years': 10, '9 years': 9, '8 years': 8, '7 years': 7, '6 years': 6, '5 years': 5, '4 years': 4, '3 years': 3,
    '2 years': 2, '1 year': 1, '< 1 year': 0
}
train.years_in_current_job = train.years_in_current_job.map(years_in_current_job_map)
test.years_in_current_job = test.years_in_current_job.map(years_in_current_job_map)

# As we know here, there are certain values as NaN, so we need to fix it
train.years_in_current_job.unique()
test.years_in_current_job.unique()

train.years_in_current_job = train.years_in_current_job.agg(lambda x: x.fillna(x.median()))
test.years_in_current_job = test.years_in_current_job.agg(lambda x: x.fillna(x.median()))

# Create a list of all values in Purpose column of train dataset
train_purpose = [i for i in train.purpose]

# Create a list of all values in Purpose column of test dataset
test_purpose = [i for i in test.purpose]

# Substract the values in test from values in train.

bad_label_cols = list(set(train_purpose) - set(test_purpose))

# Remove renewable energy
train = train[train.purpose != 'renewable energy']

# Proceed to fulfill null values with its column median
train.bankruptcies = train.bankruptcies.agg(lambda x: x.fillna(x.median()))
train.months_since_last_delinquent = train.months_since_last_delinquent.agg(lambda x: x.fillna(x.median()))
train.credit_score = train.credit_score.agg(lambda x: x.fillna(x.median()))
train.annual_income = train.annual_income.agg(lambda x: x.fillna(x.mean()))

test.bankruptcies = test.bankruptcies.agg(lambda x: x.fillna(x.median()))
test.months_since_last_delinquent = test.months_since_last_delinquent.agg(lambda x: x.fillna(x.median()))
test.credit_score = test.credit_score.agg(lambda x: x.fillna(x.median()))
test.annual_income = test.annual_income.agg(lambda x: x.fillna(x.mean()))

# Feature Engineering
corr = train.corr()
sns.heatmap(train.corr(),
             xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

st.write(sns)
