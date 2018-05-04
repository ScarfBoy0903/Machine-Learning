import numpy as np
import csv
import sys

test_path = sys.argv[1]
predict_path = sys.argv[2]

# function define
feature_name = ['AMB_TEMP', 'CH4', 'CO', 'NMHC', 'NO', 'NO2', 'NOx',
                'O3', 'PM10', 'PM2.5', 'RAINFALL', 'RH', 'SO2', 'THC',
                'WD_HR', 'WIND_DIREC', 'WIND_SPEED', 'WS_HR']

feature_idx = {'AMB_TEMP': 0, 'CH4': 1, 'CO': 2, 'NMHC': 3, 'NO': 4, 'NO2': 5, 'NOx': 6,
               'O3': 7, 'PM10': 8, 'PM2.5': 9, 'RAINFALL': 10, 'RH': 11, 'SO2': 12, 'THC': 13,
               'WD_HR': 14, 'WIND_DIREC': 15, 'WIND_SPEED': 16, 'WS_HR': 17}
def clean_test(data):
    for i in range(len(data)):
        for j in range(len(data[0])):
            if (data[i,j] < 0):
                if (j == 0):
                    data[i,j] = 0
                else:
                    data[i,j] = data[i,j-1]
    return data

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


# read model
w = np.load('model.npy')

test_x = []
n_row = 0
test_file = open(test_path, 'r')
row = csv.reader(test_file , delimiter= ",")

for r in row:
    if n_row %18 == 0:
        test_x.append([])
        for i in range(2,11):
            test_x[-1].append(float(r[i]))
    else :
        for i in range(2,11):
            if r[i] != 'NR':
                test_x[-1].append(float(r[i]))
            else:
                test_x[-1].append(0)
    n_row = n_row+1

test_x = np.array(test_x)

# clean test
test_x = clean_test(test_x)

# convert WD_value
test_x[:,14*9:15*9+9] = np.sin(test_x[:,14*9:15*9+9])

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

# filt unused feature
pick_list = np.array([['CO',1],['NMHC',4],['O3',4],['PM10',1],['PM2.5',1],
                      ['RAINFALL',3],['SO2',1],['WD_HR',1]])
test_x = pick_feature(test_x,pick_list)

ans = []
for i in range(len(test_x)):
    ans.append(["id_"+str(i)])
    a = np.dot(w,test_x[i])
    ans[i].append(a)

predict_file = open(predict_path, 'w')
s = csv.writer(predict_file, delimiter=',',lineterminator='\n')
s.writerow(["id","value"])
for i in range(len(ans)):
    s.writerow(ans[i])

test_file.close()
predict_file.close()
