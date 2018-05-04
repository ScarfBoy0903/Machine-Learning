from keras.models import load_model
import numpy as np
import pandas as pd
import math
import csv
import sys

# load model
model = load_model('best_model.h5')

# read test data
test_df = pd.read_csv(sys.argv[1])
test_str = np.array(test_df['feature'].values)
size = int(math.sqrt(len(test_str[0].split())))
length = len(test_str)
test_data = np.empty((length, size*size))
test_data_128 = np.empty((length, size*size))

for i in range(0, len(test_str)):
    test_data[i] = np.array(test_str[i].split(),dtype = 'float64')/255
    test_data_128[i] = (np.array(test_str[i].split(),dtype = 'float64')-127.5)/255
test_data = test_data.reshape(-1, size, size, 1)
test_data_128 = test_data_128.reshape(-1, size, size, 1)

pre_label = model.predict([test_data,test_data_128])

ans = np.empty(len(pre_label))
for i in range(len(pre_label)):
    ans[i] = pre_label[i].argmax()

# output the csv
text = open(sys.argv[2], "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(ans)):
    s.writerow([str(i), int(ans[i])])
text.close()
