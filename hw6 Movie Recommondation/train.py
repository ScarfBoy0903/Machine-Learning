# coding: utf-8
import numpy as np
import pandas as pd
import numpy as np
from keras.utils.generic_utils import get_custom_objects
import keras.backend as K
from keras.utils import np_utils
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Embedding, Dropout, Bidirectional, Input, BatchNormalization, Activation, Merge, Dot, Reshape, Add, Concatenate, Flatten 
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, History
from keras.regularizers import l2
import csv

np.random.seed(5566)

def rmse(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_true*rating_std - y_pred*rating_std)))

class Swish(Activation):
    
    def __init__(self, activation, **kwargs):
        super(Swish, self).__init__(activation, **kwargs)
        self.__name__ = 'swish'

def swish(x):
    return (K.sigmoid(x) * x)

get_custom_objects().update({'swish': Swish(swish)})

def DNN_model(embed_dim, reg):
    user_id = Input(shape=(1,))
    user_model = Sequential()
    user_model.add(Embedding(user_num, embed_dim, input_length=1, embeddings_regularizer = l2(reg)))
    user_model.add(Flatten())

    movie_id = Input(shape=(1,))
    movie_model = Sequential()
    movie_model.add(Embedding(movie_num, embed_dim, input_length=1, embeddings_regularizer = l2(reg)))
    movie_model.add(Flatten())

    user_vec = user_model(user_id)
    movie_vec = movie_model(movie_id) 

    concat = Concatenate()([user_vec, movie_vec])

    out = BatchNormalization()(concat)
    out = Dropout(0.3)(out)
    out = Dense(int(embed_dim), activation=swish)(out)
    out = Dropout(0.3)(out)
    
    score = Dense(1, activation='linear')(out)
    DN_model = Model(inputs = [user_id, movie_id], outputs = score)
    DN_model.compile(loss='mse', optimizer= 'adam', metrics=[rmse])
    
    return DN_model

def MF_model(embed_dim, reg):
    user_id = Input(shape=(1,))
    user_model = Sequential()    
    user_model.add(Embedding(user_num, embed_dim, input_length=1, embeddings_regularizer = l2(reg)))
    user_model.add(Flatten())
    
    movie_id = Input(shape=(1,))
    movie_model = Sequential()
    movie_model.add(Embedding(movie_num, embed_dim, input_length=1, embeddings_regularizer = l2(reg)))
    movie_model.add(Flatten())

    user_vec = user_model(user_id)
    movie_vec = movie_model(movie_id) 
    
    user_bias = Embedding(user_num, 1, embeddings_regularizer = l2(reg))(user_id)
    user_bias = Flatten()(user_bias)
    
    movie_bias = Embedding(movie_num, 1, embeddings_regularizer = l2(reg))(movie_id)
    movie_bias = Flatten()(movie_bias)
    
    score = Add()([Dot(axes=-1)([user_vec, movie_vec]), user_bias, movie_bias])
    
    MF_model = Model(inputs = [user_id, movie_id], outputs = score)
    MF_model.compile(loss= 'mse', optimizer= 'adam', metrics=[rmse])

    return MF_model

def pro_model(u_matrix, m_matrix, embed_dim, reg):
    user_id = Input(shape=(1,))
    user_model = Sequential()    
    user_model.add(Embedding(user_num, embed_dim, input_length=1, embeddings_regularizer = l2(reg)))
    user_model.add(Flatten())
    
    movie_id = Input(shape=(1,))
    movie_model = Sequential()
    movie_model.add(Embedding(movie_num, embed_dim, input_length=1, embeddings_regularizer = l2(reg)))
    movie_model.add(Flatten())

    user_vec = user_model(user_id)
    movie_vec = movie_model(movie_id) 
    
    user_pros = Sequential()    
    user_pros.add(Embedding(user_num, u_matrix.shape[1], input_length=1, weights=[u_matrix], trainable=False, embeddings_regularizer = l2(reg)))
    user_pros.add(Flatten())
    user_pros.add(Dense(units = 1, activation=swish))
    u_pros = user_pros(user_id)
    user_vec = Add()([user_vec, u_pros])
    
    movie_pros = Sequential()   
    movie_pros.add(Embedding(movie_num, m_matrix.shape[1], input_length=1, weights=[m_matrix], trainable=False, embeddings_regularizer = l2(reg)))
    movie_pros.add(Flatten())
    movie_pros.add(Dense(units = 1, activation=swish))
    m_pros = movie_pros(movie_id)
    movie_vec = Add()([movie_vec, m_pros])
    
    user_bias = Embedding(user_num, 1, embeddings_regularizer = l2(reg))(user_id)
    user_bias = Flatten()(user_bias)
    
    movie_bias = Embedding(movie_num, 1, embeddings_regularizer = l2(reg))(movie_id)
    movie_bias = Flatten()(movie_bias)
    
    
    score = Add()([Dot(axes=-1)([user_vec, movie_vec]), user_bias, movie_bias])
    
    P_model = Model(inputs = [user_id, movie_id], outputs = score)
    P_model.compile(loss= 'mse', optimizer= 'adam', metrics=[rmse])

    return P_model

train = pd.read_csv('train.csv')
user_num = train['UserID'].drop_duplicates().max() + 1
movie_num = train['MovieID'].drop_duplicates().max() + 1

rating_std = train['Rating'].values.std()
rating_mean = train['Rating'].values.mean()

user_df = pd.read_csv('users.csv').values
user_data = []
for row in user_df:
    user_data.append(row[0].split('::'))
user_data = np.array(user_data)

user_matrix = np.zeros((len(user_data) + 1,4))
for user in user_data:
    ID = int(user[0])
    if(user[1] == 'M'):
        user_matrix[ID,0] = 1
    elif(user[1] == 'F'):
        user_matrix[ID,1] = 1
    user_matrix[ID,2] = user[2]
    user_matrix[ID,3] = user[3]

Age_std = user_matrix[:,2].std()
Age_mean = user_matrix[:,2].mean()
user_matrix[:,2] = (user_matrix[:,2] - Age_mean) / Age_std

Ocu_std = user_matrix[:,3].std()
Ocu_mean = user_matrix[:,3].mean()
user_matrix[:,3] = (user_matrix[:,3] - Ocu_mean) / Ocu_std

def filt_word(word):
    s = ""
    for a in word:
        if((ord(a) >= 65 and ord(a) <= 90) or (ord(a) >= 97 and ord(a) <= 122)):
            s = s + a
    return s

fp = open('movies.csv','rb')
movie_data = []
movie_ID = []
for line in fp:
    tmp = str(line.strip()).split('::')[2].split('|')
    
    d = str((line.strip())).split('::')[0]
    di = ""
    for c in d:
        if(c.isdigit()):
            di = di + c
    if(di.isdigit()):
        movie_ID.append(int(di))
    sen = []
    for word in tmp:
        sen.append(filt_word(word)) 
    movie_data.append(sen)

movie_matrix = np.zeros((movie_num, 18))
i = 0
for cates in movie_data[1:]:
    ID = movie_ID[i]
    for cate in cates:
        if (cate == 'Fantasy'):
            movie_matrix[ID,0] = 1 
        elif (cate == 'Animation'):
            movie_matrix[ID,1] = 1 
        elif (cate == 'Childrens'):
            movie_matrix[ID,2] = 1 
        elif (cate == 'Comedy'):
            movie_matrix[ID,3] = 1 
        elif (cate == 'Adventure'):
            movie_matrix[ID,4] = 1 
        elif (cate == 'Romance'):
            movie_matrix[ID,5] = 1 
        elif (cate == 'Drama'):
            movie_matrix[ID,6] = 1 
        elif (cate == 'Action'):
            movie_matrix[ID,7] = 1 
        elif (cate == 'Crime'):
            movie_matrix[ID,8] = 1 
        elif (cate == 'Thriller'):
            movie_matrix[ID,9] = 1 
        elif (cate == 'Horror'):
            movie_matrix[ID,10] = 1 
        elif (cate == 'SciFi'):
            movie_matrix[ID,11] = 1 
        elif (cate == 'Documentary'):
            movie_matrix[ID,12] = 1 
        elif (cate == 'War'):
            movie_matrix[ID,13] = 1 
        elif (cate == 'Musical'):
            movie_matrix[ID,14] = 1 
        elif (cate == 'Mystery'):
            movie_matrix[ID,15] = 1 
        elif (cate == 'FilmNoir'):
            movie_matrix[ID,16] = 1 
        elif (cate == 'Western'):
            movie_matrix[ID,17] = 1 
        else:
            print(cate)
    i += 1
for i in range(10):
    train = train.sample(frac=1., random_state = 5566)

user = train['UserID'].values
movie = train['MovieID'].values
rating_org = train['Rating'].values
rating = (rating_org - rating_mean) / rating_std

BATCH_SIZE = 1024
NUM_EPOCHS = 100

model = MF_model(100,0.00001)
earlystoper = EarlyStopping(monitor="val_rmse", patience = 2)
filep= "MF_model.h5"
checkpointer = ModelCheckpoint(monitor='val_rmse', filepath=filep, verbose=1, save_best_only=True)

train_history = model.fit([user, movie], 
                          rating, batch_size = BATCH_SIZE, 
                          epochs = NUM_EPOCHS, 
                          validation_split = 0.1,
                          callbacks = [checkpointer, earlystoper])
