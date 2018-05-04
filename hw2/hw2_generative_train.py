import numpy as np
import pandas as pd
import math
import csv

# read train data by dataframe
x = np.array(pd.read_csv('train_X').values, dtype='float64')

# read output data by dataframe
y = np.array(pd.read_csv('train_Y', header = None), dtype='float64')

mean = np.zeros(len(x[0]))
std = np.zeros(len(x[0]))

for i in range(len(x[0])):
    mean[i] = np.mean(x[:,i]) 
    std[i] = np.std(x[:,i])
    x[:,i] = (x[:,i] - mean[i])/std[i]
    
is_data = x[np.where(y[:, 0] == 1)]
no_data = x[np.where(y[:, 0] == 0)]

cnt1 = len(is_data)
cnt2 = len(no_data)

mu1 = is_data.mean(axis=0)
mu2 = no_data.mean(axis=0)

sigma1 = np.zeros((len(x[0]),len(x[0])))
sigma2 = np.zeros((len(x[0]),len(x[0])))

for i in range(len(x)):
    if(y[i] == 1):
        sigma1 += np.dot(np.transpose([x[i] - mu1]), [(x[i] - mu1)])
    else:
        sigma2 += np.dot(np.transpose([x[i] - mu2]), [(x[i] - mu2)])
sigma1 /= cnt1
sigma2 /= cnt2

shared_sigma = (float(cnt1) / len(x)) * sigma1 + (float(cnt2) / len(x)) * sigma2
N1 = cnt1
N2 = cnt2

sigma_inverse = np.linalg.inv(shared_sigma)
w = np.dot( (mu1-mu2), sigma_inverse)
b = (-0.5) * np.dot(np.dot([mu1], sigma_inverse), mu1) + (0.5) * np.dot(np.dot([mu2], sigma_inverse), mu2) + np.log(float(N1)/N2)

# set model
np.save('generative_weight.npy', w)
np.save('generative_bias.npy', b)
np.save('generative_mean.npy', mean)
np.save('generative_std.npy', std)
