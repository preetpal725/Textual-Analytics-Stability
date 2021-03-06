#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 22:46:46 2019

@author: jiangzhaobo
"""

import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers.recurrent import LSTM, GRU
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from sklearn import preprocessing, decomposition, model_selection, metrics, pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from keras.layers import GlobalMaxPooling1D, Conv1D, MaxPooling1D, Flatten, Bidirectional, SpatialDropout1D
from keras.preprocessing import sequence, text
from keras.callbacks import EarlyStopping
from nltk import word_tokenize
import jieba
data=pd.read_csv('stability.csv') 
X=data['Sentences']
X=[i.split() for i in X]
X[:2]
import gensim
model = gensim.models.Word2Vec(X,min_count =5,window =8,size=100)   
embeddings_index = dict(zip(model.wv.index2word, model.wv.vectors))

print('Found %s word vectors.' % len(embeddings_index))
model['stability']
def sent2vec(s):
    import jieba
    words = str(s).lower()
    words = jieba.lcut(words)
#    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(embeddings_index[w])
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    if type(v) != np.ndarray:
        return np.zeros(300)
    return v / np.sqrt((v ** 2).sum())
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.Classification.values)
xtrain, xvalid, ytrain, yvalid = train_test_split(data.Sentences.values, y, stratify=y, random_state=42, test_size=0.1, shuffle=True)
xtrain_w2v = [sent2vec(x) for x in tqdm(xtrain)]
xvalid_w2v = [sent2vec(x) for x in tqdm(xvalid)]

xtrain_w2v = np.array(xtrain_w2v)
xvalid_w2v = np.array(xvalid_w2v)




#Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf=GaussianNB() 
clf.fit(xtrain_w2v,ytrain) 
predictions = clf.predict(xvalid_w2v)
print(predictions)
type(yvalid)
actual= np.array(yvalid.tolist())
actual
rere=0
reir=0
irre=0
irir=0
nnn=len(actual)
for ii in range(0,nnn):
    if actual[ii]==1:
        if predictions[ii]==1:
            rere+=1
        else:
            irre+=1
    else:
        if predictions[ii]==1:
            reir+=1
        else:
            irir+=1
#The accurary of Naive Bays
accur=(irir+rere)/(irir+rere+reir+irre)
print('The accuracy of Naive Bayes classifier is :',+accur)

#The relevant recall
reca=(rere)/(rere+reir)
print('The relevant recall of Naive Bayes classifier is :',+reca)
#The irrelevant recall
irca=(irir)/(irre+irir)
print('The irrelevant recall of Naive Bayes classifier is :',+irca)
#The relevant precision
repr=(rere)/(rere+irre)
print('The relevant precision of Naive Bayes classifier is :',+repr)
#The irrelevant precision
irpr=(irir)/(irir+reir)
print('The irrelevant precision of Naive Bayes classifier is :',+irpr)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(actual, predictions)
print(fpr)
print(tpr)
print(thresholds)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(actual, predictions)
print('The auc:',auc)
def class_logloss(actual, predicted, eps=1e-15):
    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
print ("logloss: %0.3f " % class_logloss(yvalid, predictions))
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.title("Naive Bayes-Stability")
#plt.title("Naive Bayes-opportunity-auc=%.4f"%(auc))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.fill_between(fpr, tpr, where=(tpr>=0), color='blue', alpha=0.5)
plt.show()
