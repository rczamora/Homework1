
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
def assetPriceReg(df_stk):
    import pandas_datareader.data as web  # module for reading datasets directly from the web
    
    # Reading in factor data
    df_factors = web.DataReader('F-F_Research_Data_5_Factors_2x3_daily', 'famafrench')[0]
    df_factors.rename(columns={'Mkt-RF': 'MKT'}, inplace=True)
    df_factors['MKT'] = df_factors['MKT']/100
    df_factors['SMB'] = df_factors['SMB']/100
    df_factors['HML'] = df_factors['HML']/100
    df_factors['RMW'] = df_factors['RMW']/100
    df_factors['CMA'] = df_factors['CMA']/100
    
    df_stock_factor = pd.merge(df_stk,df_factors,left_index=True,right_index=True) # Merging the stock and factor returns dataframes together
    df_stock_factor['XsRet'] = df_stock_factor['Returns'] - df_stock_factor['RF'] # Calculating excess returns

    # Running CAPM, FF3, and FF5 models.
    CAPM = sm.ols(formula = 'XsRet ~ MKT', data=df_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    FF3 = sm.ols( formula = 'XsRet ~ MKT + SMB + HML', data=df_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})
    FF5 = sm.ols( formula = 'XsRet ~ MKT + SMB + HML + RMW + CMA', data=df_stock_factor).fit(cov_type='HAC',cov_kwds={'maxlags':1})

    CAPMtstat = CAPM.tvalues
    FF3tstat = FF3.tvalues
    FF5tstat = FF5.tvalues

    CAPMcoeff = CAPM.params
    FF3coeff = FF3.params
    FF5coeff = FF5.params

    # DataFrame with coefficients and t-stats
    results_df = pd.DataFrame({'CAPMcoeff':CAPMcoeff,'CAPMtstat':CAPMtstat,
                               'FF3coeff':FF3coeff, 'FF3tstat':FF3tstat,
                               'FF5coeff':FF5coeff, 'FF5tstat':FF5tstat},
    index = ['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])


    dfoutput = summary_col([CAPM,FF3, FF5],stars=True,float_format='%0.4f',
                  model_names=['CAPM','FF3','FF5'],
                  info_dict={'N':lambda x: "{0:d}".format(int(x.nobs)),
                             'Adjusted R2':lambda x: "{:.4f}".format(x.rsquared_adj)}, 
                             regressor_order = ['Intercept', 'MKT', 'SMB', 'HML', 'RMW', 'CMA'])

    print(dfoutput)
    
    return results_df

# %%
from pathlib import Path
import sys
import os

home = str(Path.home())
print(home)


# %% 
if sys.platform == 'linux':
    inputDir = '/datasets/stocks/' 
elif sys.platform == 'win32':
    inputDir = 'C:\\Users\\rczamora\\datasets\\stocks' 
else :
    inputDir = '/datasets/stocks/'
    
fullDir= home+inputDir
print(fullDir)
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