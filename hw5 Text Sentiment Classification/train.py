# coding: utf-8
import numpy as np
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense, Embedding, Dropout, Bidirectional, Input, BatchNormalization, GRU, Activation  
from keras.optimizers import Adam, SGD
from keras.preprocessing import sequence  
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from collections import defaultdict, Counter
from keras.utils import to_categorical
from keras.preprocessing import sequence  
import sys
import re
import csv

label_train_path = sys.argv[1]
nolabel_train_path = sys.argv[2]

np.random.seed(5566)

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

def build_counter(texts):
    frequency = Counter()
    for text in texts:
        for token in text:
            frequency[token] += 1
    return frequency

def valid_generator(data, label):
    tmp_data = []
    tmp_label = []
    randomlist = np.random.permutation(len(data))
    for i in range(len(data)):
        idx = randomlist[i]
        tmp_data.append(data[idx])
        tmp_label.append(label[idx])
    
    val_size = int(0.2 * len(data))
    tmp_data = np.array(tmp_data)
    tmp_label = np.array(tmp_label)
    
    return tmp_data[val_size:], tmp_label[val_size:], tmp_data[:val_size], tmp_label[:val_size]

def build_w2i_map(fre_dic, threshold):
    MAX_FEATURES = threshold
    MAX_SENTENCE_LENGTH = 39

    word_freqs = fre_dic
    word2index = {'PAD':0,'UNK':1}
    word2index.update({x[0]: i+2 for i, x in enumerate(word_freqs.most_common(MAX_FEATURES))})
    index2word = {v:k for k, v in word2index.items()}
    return word2index, index2word

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

train_txt = []
label = []
fp = open(label_train_path, "r", encoding = 'utf-8')
for line in fp.readlines(): 
    label.append(int(line[0]))
    train_txt.append(filt_words(preprocess(line.strip()).split()[2:]))    
fp.close()

sentence = train_txt.copy()

fp = open(nolabel_train_path, "r", encoding = 'utf-8')
for line in fp.readlines(): 
    sentence.append(filt_words(preprocess(line.strip()).split()))
fp.close()

txt_counter = build_counter(sentence)
word2index, index2word = build_w2i_map(txt_counter, 100000)
np.save('word2index.npy',word2index)

seq = doc2seq(train_txt, word2index)
tra_x, tra_y, val_x, val_y = valid_generator(seq, to_categorical(label))

w2v_matrix = np.load('w2v_matrix.npy')

model = Sequential()
embedding_layer = Embedding(len(w2v_matrix), output_dim = 300, weights=[w2v_matrix], input_length = 39, trainable=False)
model.add(embedding_layer)
model.add(Bidirectional(LSTM(128,activation="tanh",dropout=0.3,return_sequences = True, kernel_initializer='he_uniform')))
model.add(Bidirectional(LSTM(128,activation="tanh",dropout=0.3,return_sequences = False, kernel_initializer='he_uniform')))
model.add(Dense(2, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

BATCH_SIZE = 64
NUM_EPOCHS = 15
earlystoper = EarlyStopping(monitor="val_loss", patience = 3)
train_history = model.fit(tra_x, tra_y, batch_size = BATCH_SIZE, epochs = NUM_EPOCHS, validation_data = (val_x, val_y), callbacks=[earlystoper])

model.save('best_model.h5')
