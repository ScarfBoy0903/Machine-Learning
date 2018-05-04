from keras.models import load_model
from sklearn import cluster
import numpy as np
import pandas as pd
import csv
import sys

# load autoencoder model
model = load_model('model.h5')

# read image file
X = np.array((np.load(sys.argv[1])),dtype = 'float64')
X = X / 255
intermediate_output = model.predict(X)

# cluster processing
clf = cluster.KMeans(init='k-means++', n_clusters = 2, random_state = 5566)
clf.fit(intermediate_output)

# read test data
test = (np.array(pd.read_csv(sys.argv[2])))[:,1:]
ans = np.zeros(test.shape[0])
for i in range(test.shape[0]):
    idx1 = test[i,0]
    idx2 = test[i,1]
    if(clf.labels_[idx1] == clf.labels_[idx2]):
        ans[i] = 1

# output the csv
text = open(sys.argv[3], "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["ID", "Ans"])
for i in range(len(ans)):
    s.writerow([str(i), int(ans[i])])
text.close()