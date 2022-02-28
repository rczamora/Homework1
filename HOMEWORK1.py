# -*- coding: utf-8 -*-
"""
Created on Wed Feb  9 12:03:26 2022

@author: rczamora
"""
# %%
import pandas as pd
import numpy as np 

import statsmodels.formula.api as sm # module for stats models
from statsmodels.iolib.summary2 import summary_col # module for presenting stats models outputs nicely


from pathlib import Path
import sys
import os

# %%
def price2ret(prices,retType='simple'):
    if retType == 'simple':
        ret = (prices/prices.shift(1))-1
    else:
        ret = np.log(prices/prices.shift(1))
    return ret

home = str(Path.home())
print(home)


# %%


if sys.platform == 'linux':
    inputDir = '/datasets/stocks/' 
elif sys.platform == 'win32':
    inputDir = '\\datasets\stocks\\' 
else :
    inputDir = '/datasets/stocks/'
    
print(sys.platform)
print(inputDir)


# %%
def price2ret(prices,retType='simple'):
    if retType == 'simple':
        ret = (prices/prices.shift(1))-1
    else:
        ret = np.log(prices/prices.shift(1))
    return ret

# %%

tgt = pd.read_csv("\\\\uem.walton.uark.edu\\UEMProfiles_Lab$\\rczamora\\RedirectedFolders\\Documents\\Finn510\\TGT1.csv",index_col='Date',parse_dates=True)

df_stk = tgt

# %%

tsla = pd.read_csv("\\\\uem.walton.uark.edu\\UEMProfiles_Lab$\\rczamora\\RedirectedFolders\\Documents\\Finn510\\TSLA1.csv",index_col='Date',parse_dates=True)

df_stk2 = tsla

# %%

wmt = pd.read_csv("\\\\uem.walton.uark.edu\\UEMProfiles_Lab$\\rczamora\\RedirectedFolders\\Documents\\Finn510\\WMT1.csv",index_col='Date',parse_dates=True)

df_stk3 = wmt

# %%

Wtgt = pd.read_csv("\\\\uem.walton.uark.edu\\UEMProfiles_Lab$\\rczamora\\RedirectedFolders\\Documents\\Finn510\\TGT1WRDS.csv",index_col='Date',parse_dates=True)

df_stk4 = Wtgt

# %%

Wtsla = pd.read_csv("\\\\uem.walton.uark.edu\\UEMProfiles_Lab$\\rczamora\\RedirectedFolders\\Documents\\Finn510\\TSLA1WRDSS.csv",index_col='Date',parse_dates=True)

df_stk5 = Wtsla

# %%

Wwmt = pd.read_csv("\\\\uem.walton.uark.edu\\UEMProfiles_Lab$\\rczamora\\RedirectedFolders\\Documents\\Finn510\\WMT1WRDSS.csv",index_col='Date',parse_dates=True)

df_stk6 = Wwmt
# %%
df_stk['Returns'] = price2ret(df_stk[['Adj Close']])
df_stk = df_stk.dropna()
df_stk.head()


# %%

"ThisPC\\Documents\\Finn510\\TGT.csv"