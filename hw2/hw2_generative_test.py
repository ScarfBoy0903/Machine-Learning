import numpy as np
import pandas as pd
import csv
import sys

weight = np.load('generative_weight.npy')
b = np.load('generative_bias.npy')
mean = np.load('generative_mean.npy')
std = np.load('generative_std.npy')

# define simoid function
def simoid(z):
    for i in range(len(z)):
        if z[i] > 500:
            z[i] = 500
        elif z[i] < -500:
            z[i] = -500
    return (1 / (1 + np.exp(-z)))

# read test data by dataframe
test_data = np.array(pd.read_csv(sys.argv[5]).values, dtype='float64')
for i in range(len(test_data[0])):
    test_data[:,i] = (test_data[:,i] - mean[i])/std[i]

test_output = np.around(simoid(np.dot(test_data, weight) + b))

# output the csv
text = open(sys.argv[6], "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(test_output)):
    s.writerow([str(i + 1), int(test_output[i])])
text.close()

