# -*- coding: utf-8 -*-
"""
Created on Sun May  9 18:28:14 2021

@author: ASHNER_NOVILLA

Please perform the code stated on Power BI and install the necessary 
Library (The Python version is 3.7 - to avoid FBProphet conflict with Power BI)

This are the dependent library with version installed:
    
Steps in installing the packages:
    1. conda create --name yourenvname python=3.7
    2. conda activate yourenvname
    3. pip install pycaret
    4. conda install pandas
    5. conda install numpy
    6. conda install matplotlib
    7. conda install seaborn
    8. conda install -c anaconda statsmodels
    9. conda install -c conda-forge fbprophet
    
Verify the packages: 
    
Package                   Version

pandas                    1.2.4     (Conda install)
matplotlib                3.4.1     (Conda install)
matplotlib-base           3.4.1     (Conda install)
numpy                     1.20.2    (Conda install)
seaborn                   0.10.1    (Conda install)
statsmodels               0.12.0    (Conda install)
fbprophet                 0.7.1     (Conda install)
pycaret                   2.3.1     (pip install)


In Running the Power BI Follow the Steps:
    1. Go to the Anaconda Promft
    2. Conda activate yourenvname     #(replace the yourenvname with the name of your environment)
    3. "C:\Program Files\Microsoft Power BI Desktop\bin\PBIDesktop.exe"
    4. Open the PowerBI file project

Change the Path Depending on the Location Folder of the Dataset
Change the destination Model for PyCaret Anomaly Detection

"""

#Change the Path Depending on the Location Folder of the Dataset

### Setting Up Needed Library ####
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import glob
import seaborn as sns
sns.set()

path = 'D:\\Disk_Drive\\DocumentsBuckUp\\360DigiCapstone\\Datasets\\'
model_path = 'D:\\Disk_Drive\\DocumentsBuckUp\\360DigiCapstone\\anomaly_deployment_051121'

total_df = pd.DataFrame()
for f in glob.glob(path+"*.xlsx"):
    df = pd.read_excel(f, 'Sheet1')
    total_df = total_df.append(df,ignore_index=True)

del (df)

total_df = total_df[['Bill Datetime', 'Patient Type', 'Maxid', 'Department Name']]

data_total = pd.DataFrame(total_df.set_index('Bill Datetime'))
data_total = pd.DataFrame(data_total['Maxid'].resample('D').nunique())
data_total['day_name'] = [i.day_name() for i in data_total.index]
data_total = data_total.rename(columns = {'Maxid': 'PatientCount'}, inplace = False).reset_index()
data_total['Bill Datetime'] =  pd.to_datetime(data_total['Bill Datetime']).dt.date


#####This is for ADFuller Test #### Try to run in PowerBI###########
from statsmodels.tsa.stattools import adfuller

X = data_total['PatientCount'].values
result = adfuller(X)
CritVal = pd.DataFrame({'ADFStat': [result[0]], 'pval':  [result[1]], "1%": [ result[4]["1%"]],
                        "5%": [ result[4]["5%"]], "10%": [ result[4]["10%"]]})

CritVal = CritVal.transpose().reset_index()
CritVal = CritVal.set_axis(['Data', 'Values'], axis=1, inplace=False)

plt.bar(CritVal['Data'], CritVal['Values'], color="green")
plt.title("AdFuller Data Analysis")
plt.text(0.85, 0.85, 'test statistic is less than the critical value, we can reject the null hypothesis', horizontalalignment='center', 
         verticalalignment='baseline', bbox=dict(facecolor='pink', alpha=0.5), fontsize=10)
plt.show()

if result[0] < result[4]["5%"]:
    print ("Reject Ho - Time Series is Stationary")
    plt.text(0.5, 0.5, 'Reject Ho: \n Time Series is Stationary', horizontalalignment='center', 
         verticalalignment='center', bbox=dict(facecolor='green', alpha=0.5), fontsize=25)
else:
    print ("Failed to Reject Ho - Time Series is Non-Stationary")
    plt.text(0.5, 0.5, 'Failed to Reject Ho: \n Time Series is Non-Stationary', horizontalalignment='center', 
         verticalalignment='center', bbox=dict(facecolor='red', alpha=0.5), fontsize=23)
#####This is for ADFuller Test #### Try to run in PowerBI###########


#####This is for statsmodels Test #### Try to run in PowerBI###########
#Checking the seasonality and trend
# import statsmodels.tsa.seasonal as stss
from statsmodels.tsa.seasonal import seasonal_decompose

s_dec_additive = seasonal_decompose(data_total['PatientCount'], period=30, model="additive")
s_dec_additive.plot()

s_dec_multiplicative = seasonal_decompose(data_total['PatientCount'], period=30, model="multiplicative")
s_dec_multiplicative.plot()

import statsmodels.graphics.tsaplots as sgt
#ACF (Auto Correlation Function) Application
sgt.plot_acf(data_total['PatientCount'], lags = 30, zero=False)  #last 40 Periods before the current one, zero to check high lags only
plt.title("ACF & S&P", size=7)
plt.show()

## Plotting the PACF
sgt.plot_pacf(data_total['PatientCount'], lags = 30, zero=False, method='ols')  #last 40 Periods before the current one, zero to check high lags only
plt.title("PACF & S&P", size=7)
plt.show()
#####This is for statsmodels Test #### Try to run in PowerBI###########


#####This is PyCaret Test #### 
## Setup environment (This method is creating a model then deploying the model in the production)
    # Please note this model is complicated as we need to 1st create a model, save the model, load the model
    # and use the mode for prediction - But in return we will have a better and detailed model.
from pycaret.anomaly import *
pycaret_s = setup(data_total[['Bill Datetime', 'PatientCount']], session_id = 123)

# # check list of available models
# models()
    
# train model
iforest_pycaret = create_model('iforest', fraction = 0.1)

#Save Model
# save_model(iforest_pycaret, 'D:/Disk_Drive/DocumentsBuckUp/360DigiCapstone/anomaly_deployment_051121') 

# ##### Try to run in PowerBI ###########
#Load Model
iforest_load = load_model(model_path)

#Predict Model
iforest_predict = predict_model(iforest_load, data = data_total)

# check anomalies
iforest_results = assign_model(iforest_pycaret)
iforest_anomaly = iforest_results[iforest_results['Anomaly'] == 1]

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
plt.plot(iforest_results['Bill Datetime'], iforest_results['PatientCount'])
plt.scatter(iforest_anomaly['Bill Datetime'], iforest_anomaly['PatientCount'], color="red", marker="x")
for i,j in zip(iforest_anomaly['Bill Datetime'], iforest_anomaly['PatientCount']):
    ax.annotate('(%s,' %i, xy=(i,j))
plt.show()

## This Model is the simplest process for looking for anomaly as we don't need to create a model anymore
    #but in return our model is not that detailed.
    
dataset = get_outliers(data = data_total)
dataset_anomaly = dataset[dataset['Anomaly'] == 1]

fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(111)
plt.plot(dataset['Bill Datetime'], dataset['PatientCount'])
plt.scatter(dataset_anomaly['Bill Datetime'], dataset_anomaly['PatientCount'], color="red", marker="x")
for i,j in zip(dataset_anomaly['Bill Datetime'], dataset_anomaly['PatientCount']):
    ax.annotate('(%s,' %i, xy=(i,j))   
plt.show()

##Getting the Mean of Each Date##
## Getiing the average for each day
data_total_sunday = data_total[data_total['day_name']=='Sunday'].mean()
data_total_monday = data_total[data_total['day_name']=='Monday'].mean()
data_total_tuesday = data_total[data_total['day_name']=='Tuesday'].mean()
data_total_wednesday = data_total[data_total['day_name']=='Wednesday'].mean()
data_total_thursday = data_total[data_total['day_name']=='Thursday'].mean()
data_total_friday = data_total[data_total['day_name']=='Friday'].mean()
data_total_saturday = data_total[data_total['day_name']=='Saturday'].mean()

## Performing Imputation on the Dataset. Changing the Holidays into regular days.
    #Take the mean of the whole data set per day and replace the holiday depending on the name of the day
data_new_date = data_total.copy()
data_new_date['Bill Datetime'] = pd.to_datetime(data_new_date['Bill Datetime'])

data_new_date['PatientCount'] = np.where((data_new_date['Bill Datetime'] == '2021-01-26 00:00:00'),data_total_tuesday, data_new_date['PatientCount']) #Republic Day
data_new_date['PatientCount'] = np.where((data_new_date['Bill Datetime'] == '2021-01-14 00:00:00'),data_total_thursday, data_new_date['PatientCount']) #Pongal
data_new_date['PatientCount'] = np.where((data_new_date['Bill Datetime'] == '2021-01-28 00:00:00'),data_total_thursday, data_new_date['PatientCount']) #Outlier
data_new_date['PatientCount'] = np.where((data_new_date['Bill Datetime'] == '2020-12-25 00:00:00'),data_total_friday, data_new_date['PatientCount']) #Christmas Day
data_new_date['PatientCount'] = np.where((data_new_date['Bill Datetime'] == '2021-01-01 00:00:00'),data_total_friday, data_new_date['PatientCount']) #New Year Day
data_new_date['PatientCount'] = np.where((data_new_date['Bill Datetime'] == '2020-11-14 00:00:00'),data_total_saturday, data_new_date['PatientCount']) #Diwali


############ FB Prophet Setup ############
from statsmodels.tools.eval_measures import rmse
from statsmodels.tools.eval_measures import mse
def mean_absolute_percentage_error(y_true, y_pred): 
    mape = np.mean(np.abs((y_true - y_pred)/y_true))*100
    return mape

from fbprophet import Prophet
model = Prophet()

train_test_var = int((len(data_new_date) * 0.75))
train_df = data_new_date.iloc[0:train_test_var,0:2]
train_df.columns = ['ds', 'y']
test_df = data_new_date.iloc[train_test_var::,0:2]
test_df.columns = ['ds', 'y']

# fit the model
model.fit(train_df)
forecast_future = model.make_future_dataframe(periods=36, freq='D', include_history=False) #Change Parameter in the Power BI
forecast = model.predict(forecast_future)
forecast = forecast[['ds','yhat']]

plt.plot(test_df['ds'], test_df['y'])
plt.plot(forecast['ds'], forecast['yhat'])
plt.title("FBProphet")
plt.show()

predicted_values = pd.merge(test_df, forecast, on='ds', how='right')
prophet_rmse = rmse(predicted_values['y'], predicted_values['yhat'])
prophet_mape = mean_absolute_percentage_error (predicted_values['y'], predicted_values['yhat'])
prophet_mse = mse(predicted_values['y'], predicted_values['yhat'])

predicted_values = predicted_values.dropna()

print("prophet_mape: ", prophet_mape)

######################################################################################

# ARIMA example
from statsmodels.tsa.arima.model import ARIMA
# fit model
arima_model = ARIMA(train_df['y'], order=(7, 1, 7))
arima_model_fit = arima_model.fit()
# make prediction
arima_yhat = arima_model_fit.predict(len(train_df['y']), len(train_df['y'])+len(test_df['y'])-1)
print(arima_yhat)

plt.plot(test_df['y'])
plt.plot(arima_yhat)

arima_mape = mean_absolute_percentage_error (test_df['y'], arima_yhat)

print("arima_mape: ", arima_mape)


#Applying the Auto Regression
from statsmodels.tsa.ar_model import AutoReg

new_AutoReg_model = train_df.copy()
new_AutoReg_model = new_AutoReg_model.set_index('ds')

new_AutoReg_model_test = test_df.copy()
new_AutoReg_model_test = new_AutoReg_model_test.set_index('ds')

AutoReg_model = AutoReg(new_AutoReg_model['y'], lags=14)
AutoReg_model_fit = AutoReg_model.fit()
# make prediction
AutoReg_yhat = AutoReg_model_fit.predict(len(train_df['y']), len(train_df['y'])+len(test_df['y'])-1)
print(AutoReg_yhat)

AutoReg_yhat = pd.DataFrame(AutoReg_yhat.reset_index())
AutoReg_yhat.columns = ['ds', 'yhat']

plt.plot(test_df['ds'], test_df['y'])
plt.plot(AutoReg_yhat['ds'], AutoReg_yhat['yhat'])
plt.title("Auto Regression")
plt.show()

# AutoReg_yhat = AutoReg_yhat.squeeze()

autoreg_predicted_values = pd.merge(test_df, AutoReg_yhat, on='ds', how='right')

autoreg_mape = mean_absolute_percentage_error (autoreg_predicted_values['y'], AutoReg_yhat['yhat'])

print("autoreg_mape: ", autoreg_mape)


