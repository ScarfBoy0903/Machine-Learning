import numpy as np
import pandas as pd
import csv
import sys

def simoid(z):
    return (1 / (1 + np.exp(-z)))

# read model
w = np.load('weight.npy')
mean = np.load('mean.npy')
std = np.load('std.npy')

# read test data
test_x = np.array(pd.read_csv(sys.argv[5]).values, dtype='float64')

# feature scaling
for i in range(len(test_x[0])):
    test_x[:,i] = (test_x[:,i] - mean[i])/std[i]

# add bias
test_x = np.concatenate((np.ones((test_x.shape[0],1)),test_x), axis=1)

test_output = np.around(simoid(np.dot(test_x, w)))

# output the csv
text = open(sys.argv[6], "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(test_output)):
    s.writerow([str(i + 1), int(test_output[i])])
text.close()


