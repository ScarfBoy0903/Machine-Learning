import numpy as np
import pandas as pd
import numpy as np
from keras.models import load_model
import keras.backend as K
import csv
import sys

test_path = sys.argv[1]
pre_destination = sys.argv[2]

rating_std = 1.116897661146206
rating_mean = 3.5817120860388076

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true*rating_std - y_pred*rating_std)))

model = load_model('MF_model.h5', custom_objects={'rmse': rmse})

test = pd.read_csv(test_path)
test_u = test['UserID'].values
test_m = test['MovieID'].values
pre_score = (model.predict([test_u,test_m])) * rating_std + rating_mean
pre_score = np.clip(pre_score, 1, 5)
ans = np.empty(len(pre_score))
for i in range(len(pre_score)):
    ans[i] = pre_score[i]

text = open(pre_destination, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["TestDataID", "Rating"])
for i in range(len(ans)):
    s.writerow([str(i + 1), (ans[i])])
text.close()
