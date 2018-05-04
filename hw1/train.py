# package import
import csv 
import numpy as np
import math
import pandas as pd

# function define
feature_name = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx',
                'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC',
                'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']

feature_idx = {'AMB_TEMP': 0, 'CH4': 1, 'CO': 2, 'NMHC': 3, 'NO': 4, 'NO2': 5, 'NOx': 6,
               'O3': 7, 'PM10': 8, 'PM2.5': 9, 'RAINFALL': 10, 'RH': 11, 'SO2': 12, 'THC': 13,
               'WD_HR': 14, 'WIND_DIREC': 15, 'WIND_SPEED': 16, 'WS_HR': 17}

def loss(x,y,w):
    hypo = np.empty((len(y)))
    hypo = np.dot(x,w)
    return(hypo - y)

def RMSE(los, xx):
    return math.sqrt(np.sum(los**2) / len(xx))

def clean_train(data,output):
    dirty_output = np.where(output <= 0)
    output = np.delete(output, np.s_[dirty_output], 0)
    data = np.delete(data, np.s_[dirty_output], 0)
    zero_feature = []
    for i in range(len(data)):
        for j in range(len(data[0])):
            if (data[i,j] < 0):
                if (j == 0):
                    data[i,j] = 0
                else:
                    data[i,j] = data[i,j-1]
    return np.hstack((output.reshape(len(data),1),data))

def pick_feature(data, pick_list):
    bias = data[:,0]
    feature = np.split(data[:, 1:len(data[0])], 18, axis=1)
    train = bias.reshape(len(bias), 1)
    for i in range(len(pick_list)):
        idx = int(feature_idx[pick_list[i,0]])
        hour = int(pick_list[i,1])
        temp = feature[idx]
        train = np.hstack((train,temp[:,-hour:]))
    return train

# read csv
fp = open('train.csv', "r")
df = pd.read_csv(fp)
fp.close()
df = (df.drop(df.columns[[0, 1, 2]], axis=1)).replace(['NR'], [0])
data = np.array(df.values, dtype='float64')

drop_set = np.array([])
set_leng = 18 - len(drop_set)
mon_data = np.split(data, 12, axis=0)
merge_matrix = np.empty((12, set_leng, 480))

x = np.empty((set_leng * 9))

y = np.empty((5652))
for mon in range(12):
    day_data = np.split(mon_data[mon], 20, axis=0)
    merge_matrix[mon] = np.hstack((day_data[0], day_data[1], day_data[2], day_data[3], day_data[4],
                                   day_data[5], day_data[6], day_data[7], day_data[8], day_data[9],
                                   day_data[10], day_data[11], day_data[12], day_data[13], day_data[14],
                                   day_data[15], day_data[16], day_data[17], day_data[18], day_data[19]))
    for num in range(471):
        y[num + mon*471] = merge_matrix[mon, 9 ,num+9] 
        x = np.vstack((x, merge_matrix[mon, :, num:num+9].reshape(set_leng*9)))

x = np.delete(x, 0, 0)

# clean data
clear_data = clean_train(x,y)
y = clear_data[:,0]
x = clear_data[:,1:]

# convert WD_value
x[:,14*9:15*9+9] = np.sin(x[:,14*9:15*9+9])

# add bias
x = np.concatenate((np.ones((x.shape[0],1)),x), axis=1)

# filt unused feature
pick_list = np.array([['CO',1],['NMHC',4],['O3',4],['PM10',1],['PM2.5',1],
                      ['RAINFALL',3],['SO2',1],['WD_HR',1]])
x = pick_feature(x, pick_list)

# train process
w = np.zeros(len(x[0]))
l_rate = 10
repeat = 50000
break_point = 0
stop_cretirion = 0.00001
tra_x = x
tra_y = y
tra_x_t = tra_x.transpose()
s_gra = np.zeros(len(tra_x[0]))
tra_RMSE = np.empty((repeat))
for i in range(repeat):
    gra = np.dot(tra_x_t,loss(tra_x,tra_y,w)) + 0.1 * w
    s_gra += gra**2
    ada = np.sqrt(s_gra)
    w = w - l_rate * gra/ada
    tra_RMSE[i] = RMSE(loss(tra_x,tra_y,w),tra_x)
    if (i > 100 and abs(tra_RMSE[i] - tra_RMSE[i - 1]) <= stop_cretirion):
        break_point = i
        break
    print ('iteration: %d | Cost: %f  ' % ( i,tra_RMSE[i]))
    break_point = i

# save model
np.save('model.npy',w)