import pickle
import numpy as np
import pandas as pd
import csv
import math
import sys

# function define
def gaussian(data, mean = 48.5816, deviation = 13.6402):
    scalar = 1/math.sqrt(2*math.pi*deviation*deviation)
    return(scalar*np.exp(-0.5*((data - mean)**2/(deviation * deviation))))

def simoid(z):
    return (1 / (1 + np.exp(-z)))

def AVE_loss(pre_y,y):
    return (np.abs(np.around(pre_y) - y).sum() / len(y))

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

# load scaler
scalerfile = 'scaler.sav'
scaler = pickle.load(open(scalerfile, 'rb'))

# save model
w = np.load('best_weight.npy')

# read test data by dataframe
test_df = pd.read_csv(sys.argv[5])
test_df['education_data'] = pd.read_csv(sys.argv[1])['education_num']

test_df = test_preprocess(test_df,scaler)
test_x = np.array(test_df.values, dtype = 'float64')
test_output = np.around(simoid(np.dot(test_x, w)))
# output the csv
text = open(sys.argv[6], "w+")

s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(test_output)):
    s.writerow([str(i + 1), int(test_output[i])])
text.close()
