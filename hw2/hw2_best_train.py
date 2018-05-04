# coding: utf-8
import numpy as np
import pandas as pd
import csv
import math
from sklearn.preprocessing import MinMaxScaler

# function define
def gaussian(data, mean = 48.5816, deviation = 13.6402):
    scalar = 1/math.sqrt(2*math.pi*deviation*deviation)
    return(scalar*np.exp(-0.5*((data - mean)**2/(deviation * deviation))))

def simoid(z):
    return (1 / (1 + np.exp(-z)))

def AVE_loss(pre_y,y):
    return (np.abs(np.around(pre_y) - y).sum() / len(y))

def train_preprocess(DF):
    # deal age
    DF['age_label'] = (DF['age'] < 30).apply(lambda x: 1 if x else 0)
    tmp = gaussian(DF['age'])
    
    # deal capital
    DF['gain_label1'] = (DF['capital_gain'] == 0).apply(lambda x: 1 if x else 0)
    DF['gain_label2'] = ((DF['capital_gain'] < 4000) & (DF['capital_gain'] > 0)).apply(lambda x: 1 if x else 0)
    DF['gain_label3'] = ((DF['capital_gain'] >= 4000) & (DF['capital_gain'] < 7000)).apply(lambda x: 1 if x else 0)
    DF['gain_label4'] = (DF['capital_gain'] >= 7000).apply(lambda x: 1 if x else 0)
    DF['gain_square'] = DF['capital_gain']**2
    DF['loss_label'] = (DF['capital_gain'] != 0).apply(lambda x: 1 if x else 0)
    DF['diff'] = DF['capital_gain'] - DF['capital_loss']
    DF['diff_cube'] = DF['diff']**3
    
    # deal work per hour
    DF['wph_label'] = (DF['hours_per_week'] < 30).apply(lambda x: 1 if x else 0)
    
    # deal education_num
    DF['ed_square'] = DF['education_data']**2
    
    # scaling
    columns = DF.columns
    scaler = MinMaxScaler()
    scaler.fit(DF)
    DF = scaler.transform(DF)
    DF = pd.DataFrame(data=DF, columns=columns)
    
    DF['age_gaussian'] = tmp
    DF['bias'] = np.ones(len(DF.values))
    
    return DF,scaler

def test_preprocess(DF,scaler):
    # deal age
    DF['age_label'] = (DF['age'] < 30).apply(lambda x: 1 if x else 0)
    tmp = gaussian(DF['age'])
    
    # deal capital
    DF['gain_label1'] = (DF['capital_gain'] == 0).apply(lambda x: 1 if x else 0)
    DF['gain_label2'] = ((DF['capital_gain'] < 4000) & (DF['capital_gain'] > 0)).apply(lambda x: 1 if x else 0)
    DF['gain_label3'] = ((DF['capital_gain'] >= 4000) & (DF['capital_gain'] < 7000)).apply(lambda x: 1 if x else 0)
    DF['gain_label4'] = (DF['capital_gain'] >= 7000).apply(lambda x: 1 if x else 0)
    DF['gain_square'] = DF['capital_gain']**2
    DF['loss_label'] = (DF['capital_gain'] != 0).apply(lambda x: 1 if x else 0)
    DF['diff'] = DF['capital_gain'] - DF['capital_loss']
    DF['diff_cube'] = DF['diff']**3
    
    # deal work per hour
    DF['wph_label'] = (DF['hours_per_week'] < 30).apply(lambda x: 1 if x else 0)
    
    # deal education_num
    DF['ed_square'] = DF['education_data']**2
    
    columns = DF.columns
    DF = scaler.transform(DF)
    DF = pd.DataFrame(data=DF, columns=columns)
    
    DF['age_gaussian'] = tmp
    DF['bias'] = np.ones(len(DF.values))
    return DF

# read train data
x_dataframe = pd.read_csv('train_X')
x_dataframe['education_data'] = pd.read_csv('train.csv')['education_num']
y_dataframe = pd.read_csv('train_Y', header = None)

np.random.seed(5566)
data_num = len(x_dataframe.values)
permute = np.random.permutation(data_num)

valid_id = permute[ : int(data_num * 0.1)]
train_id = permute[int(data_num * 0.1) :]

trainX = x_dataframe.iloc[train_id].copy(deep=True).reset_index(drop=True)
trainY = y_dataframe.iloc[train_id].copy(deep=True).reset_index(drop=True)
validX = x_dataframe.iloc[valid_id].copy(deep=True).reset_index(drop=True)
validY = y_dataframe.iloc[valid_id].copy(deep=True).reset_index(drop=True)

trainX,scaler = train_preprocess(trainX)
validX = test_preprocess(validX,scaler)

tra_x = np.array(trainX.values,dtype = 'float64')
tra_y = np.array(trainY.values,dtype = 'float64')
val_x = np.array(validX.values,dtype = 'float64')
val_y = np.array(validY.values,dtype = 'float64')

tra_y = tra_y.reshape(len(tra_y))
val_y = val_y.reshape(len(val_y))

repeat = 2230

alpha = 0.05
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8

tra_Loss = np.empty((repeat-1))
val_Loss = np.empty((repeat-1))

w = np.zeros(len(tra_x[0]))

tra_x_t = tra_x.transpose()

m_t = np.zeros(len(tra_x[0])) 
v_t = np.zeros(len(tra_x[0]))
g_t = np.zeros(len(tra_x[0]))

for t in range(1,repeat):
    pre_y = simoid(np.dot(tra_x, w))
    
    g_t = np.dot(tra_x_t,pre_y - tra_y)
    m_t = beta_1*m_t + (1-beta_1)*g_t
    v_t = beta_2*v_t + (1-beta_2)*(g_t**2)
    
    m_cap = m_t/(1-(beta_1**t))
    v_cap = v_t/(1-(beta_2**t))
    
    tra_Loss[t - 1] = AVE_loss(pre_y,tra_y)
    val_Loss[t - 1] = AVE_loss(simoid(np.dot(val_x,w)),val_y)
    
    print ('iteration: %d | tra_core: %f | val_core: %f ' % ( t,1-tra_Loss[t - 1],1-val_Loss[t - 1]))
    break_point = t
    
    w_prev = w
    w = w - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)

# save model
np.save('best_weight.npy',w)

import pickle
scalerfile = 'scaler.sav'
pickle.dump(scaler, open(scalerfile, 'wb'))