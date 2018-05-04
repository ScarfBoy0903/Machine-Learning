# coding: utf-8
import numpy as np
import pandas as pd
import csv

# function define
def min_max_normalize(col):
    mean = col.min()
    dev = col.max() - col.min()
    return [mean, dev, (col - mean)/dev]

def simoid(z):
    return (1 / (1 + np.exp(-z)))

def AVE_loss(pre_y,y):
    return (np.abs(np.around(pre_y) - y).sum() / len(y))

# read train data
x = np.array(pd.read_csv('train_X'), dtype='float64')
y = np.array(pd.read_csv('train_Y', header = None), dtype='float64')
y = y.reshape(len(y))

# feature scaling
mean = np.empty(len(x[0]))
std = np.empty(len(x[0]))

for i in range(len(x[0])):
    norm_data = min_max_normalize(x[:,i])
    mean[i] = norm_data[0]
    std[i] = norm_data[1]
    x[:,i] = norm_data[2]

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

repeat = 2712

alpha = 0.01
beta_1 = 0.7
beta_2 = 0.7


epsilon = 1e-8
tra_Loss = np.empty((repeat-1))
w = np.zeros(len(x[0]))

tra_x = x
tra_y = y
tra_x_t = tra_x.transpose()

m_t = np.zeros(len(tra_x[0])) 
v_t = np.zeros(len(tra_x[0]))
g_t = np.zeros(len(tra_x[0]))

for t in range(1,repeat):
    pre_y = simoid(np.dot(tra_x, w))
    
    g_t = np.dot(tra_x_t,pre_y - y) + 0.1 * w
    m_t = beta_1*m_t + (1-beta_1)*g_t
    v_t = beta_2*v_t + (1-beta_2)*(g_t**2)
    
    m_cap = m_t/(1-(beta_1**t))		#calculates the bias-corrected estimates
    v_cap = v_t/(1-(beta_2**t))		#calculates the bias-corrected estimates
    
    tra_Loss[t - 1] = AVE_loss(pre_y,y)
    print ('iteration: %d | Cost: %f  ' % ( t,tra_Loss[t - 1]))
    break_point = t
    
    w_prev = w
    w = w - (alpha*m_cap)/(np.sqrt(v_cap)+epsilon)	#updates the parameters
    
# save model
np.save('weight.npy',w)
np.save('mean.npy',mean)
np.save('std.npy',std)