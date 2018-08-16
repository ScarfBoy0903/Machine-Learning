import numpy as np
import pandas as pd
import numpy as np
import sys
import re
import csv

from keras.preprocessing import sequence  
from keras.models import load_model

test_path = sys.argv[1]
destination = sys.argv[2]  

def preprocess(string):
    for same_char in re.findall(r'((\w)\2{2,})', string):
        string = string.replace(same_char[0], same_char[1])
    for digit in re.findall(r'\d+', string):
        string = string.replace(digit, "1")
    for punct in re.findall(r'([-/\\\\()!"+,&?\'.]{2,})',string):
        if punct[0:2] =="..":
            string = string.replace(punct, "...")
        else:
            string = string.replace(punct, punct[0])
    string = string.replace("i ' m", "im").replace("you ' re","youre").replace("didn ' t","didnt").replace("can ' t","cant").replace("haven ' t", "havent").replace("won ' t", "wont")    .replace("isn ' t","isnt").replace("don ' t", "dont").replace("doesn ' t", "doesnt")    .replace("aren ' t", "arent").replace("weren ' t", "werent").replace("wouldn ' t","wouldnt")    .replace("ain ' t","aint").replace("shouldn ' t","shouldnt").replace("wasn ' t","wasnt")    .replace(" ' s","s").replace("wudn ' t","wouldnt").replace(" .. "," ... ")    .replace("couldn ' t","couldnt")
    
    return string
def doc2seq(data, word2index):
    X = np.empty(len(data),dtype=list)
    i=0
    for words in data:
        seqs = []
        for word in words:
            if word in word2index:
                seqs.append(word2index[word])
            else:
                seqs.append(word2index["UNK"])            
        X[i] = seqs
        i += 1
    X = sequence.pad_sequences(X, maxlen = 39)
    return X

def filt_sentences(data, word2index):
    doc = []
    for words in data:
        sen = []
        for word in words:
            if word in word2index:
                sen.append(word)
            else:
                sen.append('UNK')               
        doc.append(sen)
    return doc

def check_ascii(word):
    for i in word:
        if (not((ord(i) >= 97 and ord(i) <= 122) or ord(i) == 33 or ord(i) == 39 or ord(i) == 44 or ord(i) == 46 or ord(i) == 58 or ord(i) == 59 or ord(i) == 63)):
            return False
    return True

def filt_words(word_seq):
    tmp = []
    for word in word_seq:
        if (check_ascii(word)):
            tmp.append(word)
    return tmp

word2index = np.load('word2index.npy').item()
fp = open(test_path, "r", encoding = 'utf-8')
test = []
for line in fp.readlines(): 
    tmp = preprocess(line.strip()).split()
    test.append(filt_words([tmp[0].split(',')[1]] + tmp[1:]))
fp.close()

model = load_model('best_model.h5')
test_seq = doc2seq(test[1:], word2index)
test_predict = model.predict(test_seq)

ans = np.zeros(len(test_predict))
for i in range(len(test_predict)):
    ans[i] = test_predict[i].argmax()

text = open(destination, "w+")
s = csv.writer(text, delimiter=',', lineterminator='\n')
s.writerow(["id", "label"])
for i in range(len(ans)):
    s.writerow([i, int(ans[i])])
text.close()