
# coding: utf-8

# In[98]:


import pandas as pd
import nltk.data
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import gensim
from keras.models import Model
from keras.layers import Input, Embedding, Dense, Dropout, Flatten
from keras.layers import Activation
import numpy as np
import chardet
import logging
from keras.models import Model
from keras.layers import Input,Dense,Dropout,Flatten
from keras.layers import Conv1D,MaxPooling1D,AveragePooling1D
from keras.layers import LSTM,Bidirectional
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from keras.models import load_model
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, roc_auc_score, precision_score
from sklearn.model_selection import cross_val_predict
from keras import models
from keras import layers
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn import cross_validation
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding
from nltk.stem import PorterStemmer


# In[129]:


def clean_text1(dataframe):
#Cleaning text    
    dataframe["doc"] = dataframe['text']
#Remove numbers
    dataframe.loc[:,"doc"] = dataframe.doc.str.replace(r'\d+','')
#Remove floats
    dataframe.loc[:,"doc"] = dataframe.doc.astype(str).replace(r'(\d*\.?\d*)','')
#Remove Punctuation
    dataframe.loc[:,"doc"] = dataframe.doc.astype(str).apply(lambda x : " ".join(re.findall('[\w]+',x)))
#Convert to lower text
    dataframe.loc[:,"doc"] = dataframe.doc.astype(str).apply(lambda x: " ".join(x.lower() for x in x.split()))      
#Stop words removal
    stop = stopwords.words('english')
    dataframe.loc[:,'doc'] = dataframe.loc[:,'doc'].apply(lambda x: " ".join(x for x in x.split() if x not in stop))
#Reomve top 50 common words
    freq = pd.Series(' '.join(dataframe['doc']).split()).value_counts()[:50]
    freq = list(freq.index)
    dataframe.loc[:,'doc'] = dataframe.loc[:,'doc'].apply(lambda x: " ".join(x for x in x.split() if x not in freq))
#Stemming        
    st = PorterStemmer()
    dataframe.loc[:,'doc'] = dataframe.loc[:,'doc'].apply(lambda x: " ".join([st.stem(word) for word in x.split()]))

# Moving cleaned text to another column
    dataframe['cleaned_text'] = dataframe["doc"]
#Tokenization                               
    dataframe.loc[:,"doc"] = dataframe.loc[:,"doc"].apply(nltk.word_tokenize)
    
    return dataframe


# In[4]:


def w2c_model(doc,size,window,min_count,workers,iter,sg,length,numepoch):    
    model = gensim.models.Word2Vec(size=size, window=window, min_count=min_count, workers=workers,iter=iter,sg=sg)
    model.build_vocab(doc)
    model.train(doc,total_examples=length,epochs = numepoch)
    return model


# In[5]:


#Convert clean text into sequence base on word2vec
def text2seq(doc, w2v, seq_len, emb_size):
    padding = np.array([0 for __ in range(emb_size)])
    seq = []
    for token in doc[: seq_len]:
        try:
            seq.append(w2v.wv[token])
        except:
            aa = 1

    seq += [padding for __ in range(seq_len - len(seq))]
    #print(len(seq))
    return seq


# In[6]:


def one_hot(series):
    values=sorted(list(set(series)))
    return np.vstack(map(lambda x: [x==v for v in values],series))


# In[7]:


def prepro(dataframe, w2v,seq_len, emb_size, target_col):
    X = dataframe['doc'].apply(lambda doc: text2seq(doc, w2v, seq_len, emb_size))
    X = np.vstack(X).reshape(len(dataframe), seq_len, emb_size)
    y = one_hot(dataframe[target_col])
    return X, y


# In[303]:


#Model building for original approach
def model1(seq_len, emb_size, num_labels):
    input_layer = Input(shape=(seq_len,emb_size)) 
    x = LSTM(100, return_sequences = True,dropout=0.2,recurrent_dropout=0.05)(input_layer)
    x = Conv1D(64, 3, activation='relu')(x)
    x = MaxPooling1D(pool_size=2)(x) 
    x = Flatten()(x)
    output_layer = Dense(num_labels, activation='softmax')(x)
    model = Model(input_layer, output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


# In[299]:


#Model for cross valiation 
def build_model3():
    input_layer = Input(shape=(100,100)) 
    x = LSTM(100, return_sequences = True,dropout=0.2,recurrent_dropout=0.05)(input_layer)
    x = Conv1D(64, 3, activation='relu')(x) #Second Change: Changed Kernal size here from 64 to 32
    x = MaxPooling1D(pool_size=2)(x) 
    x = Flatten()(x)
#     x = Dense(256, activation='relu')(x)
#     x = Dropout(0.2)(x)                   #Fourth Change: Adding dropout layer here
#     x = Dense(64, activation='relu')(x)   #Third Change: Adding dense layer here
# #     x = Dropout(0.2)(x)
#     x = Dense(128, activation='relu')(x)
#     x = Dropout(0.2)(x)
    output_layer = Dense(9, activation='softmax')(x)
    model = Model(input_layer, output_layer)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
    return model


# In[304]:


def fit_model(X,y,weightsfile,num_labels,seq_len,emb_size,epochs):
    checkpoint = ModelCheckpoint(weightsfile, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    model_dl = model1(seq_len, emb_size, num_labels)
    model_dl.fit(X, y, epochs=epochs, batch_size=16,callbacks=callbacks_list, validation_split=0.2)
    return model_dl


# In[307]:


def prediction(model,weightsfile,X):
    model.load_weights(weightsfile)
    preds = model.predict(X)
    lst = list(map(lambda x: x.argmax(), preds))
    return preds,lst


# # Processing Data

# In[11]:


df = pd.read_csv('data.csv')


# In[12]:


df.head()


# In[130]:


#Clean the text column
df = clean_text1(df)
#Assign processed column to be the target column for the word2vec
doc = df.doc
#create a word2vec model to use as embedding layers in CNN
w2c = w2c_model(doc,size=100,window=5,min_count=1,workers=6,iter=10,sg=0,length=len(doc),numepoch=10)


# In[134]:


df.head()


# In[142]:


df = df.sample(frac=1)


# In[175]:


#Create sequence and embedding layer base on word2vec 
X, y = prepro(df,w2c,seq_len=100, emb_size=100,target_col='Label')


# In[300]:


# Wrap Keras model so it can be used by scikit-learn
CNN = KerasClassifier(build_fn=build_model3, 
                            #batch_size = 64,  #Fifth Change: batch size from 32 to 64
                            epochs=20) #First change: From 20 to 10
                                 


# In[302]:


#Use cross validation predict to make prediction on the entire dataset
#The test & train accuracy is 99.12%
cnn_pred = cross_val_predict(CNN, X, y, cv=10)


# In[291]:


#Assign the prediction back to dataframe
# df['predict_b64'] = cnn_pred


# In[180]:


# values=sorted(list(set(df['Label'])))
# values


# In[273]:


df.head()


# # Accuracy, Precision & Recall for original model

# In[316]:


print(classification_report(df['Label'],df['predict_64']))

#Accuracy score
accuracy_score(df['Label'],df['predict_64'])*100


# # Accuracy, Precision & Recall for Epochs = 10

# In[317]:


print(classification_report(df['Label'],df['predict_e10']))

#Accuracy score
accuracy_score(df['Label'],df['predict_e10'])*100


# # Accuracy, Precision & Recall for Kernal Size = 32

# In[318]:


print(classification_report(df['Label'],df['predict1']))

#Accuracy score
accuracy_score(df['Label'],df['predict1'])*100


# # Accuracy, Precision & Recall for Adding Dense Layer (64 nodes, 'relu' activation)

# In[319]:


print(classification_report(df['Label'],df['predict_d64']))

#Accuracy score
accuracy_score(df['Label'],df['predict_d64'])*100


# # Accuracy, Precision & Recall for Adding Dropout layer with dropout rate 0.2

# In[323]:


print(classification_report(df['Label'],df['predict_d0.2']))

#Accuracy score
accuracy_score(df['Label'],df['predict_d0.2'])*100


# # Accuracy, Precision & Recall for changing batch size to 64

# In[324]:


print(classification_report(df['Label'],df['predict_b64']))

#Accuracy score
accuracy_score(df['Label'],df['predict_b64'])*100


# # Training and make prediction in a traditional way

# In[306]:


#Train model
#Accuracy is 99.43%
CNN_model= fit_model(X=X,y=y,weightsfile = "first.hdf5",num_labels=9,seq_len=100,emb_size=100,epochs=20)


# In[308]:


#Make prediction on the entire dataset
preds,lst = prediction(model=CNN_model,weightsfile = "first.hdf5",X = X)


# In[309]:


#Assign the prediction values back to dataframe
df['p'] = lst


# In[325]:


print(classification_report(df['Label'],df['p']))
accuracy_score(df['Label'],df['p'])*100

