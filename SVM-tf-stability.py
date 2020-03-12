#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 22:46:46 2019

@author: jiangzhaobo
"""

import pandas as pd
import numpy as np
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

data=pd.read_csv('stability.csv') 
data.head()
data.info()
data.Classification.unique()
lbl_enc = preprocessing.LabelEncoder()
y = lbl_enc.fit_transform(data.Classification.values)
xtrain, xvalid, ytrain, yvalid = train_test_split(data.Sentences.values, y, stratify=y,random_state=42, test_size=0.1, shuffle=True)
print (xtrain.shape)
print (xvalid.shape)

def number_normalizer(tokens):
    return ("#NUMBER" if token[0].isdigit() else token for token in tokens)
class NumberNormalizingVectorizer(TfidfVectorizer):
    def build_tokenizer(self):
        tokenize = super(NumberNormalizingVectorizer, self).build_tokenizer()
        return lambda doc: list(number_normalizer(tokenize(doc)))
    
    
tfv = NumberNormalizingVectorizer(min_df=3,  
                                  max_df=0.5,
                                  max_features=None,                 
                                  ngram_range=(1, 2), 
                                  use_idf=True,
                                  smooth_idf=True)

tfv.fit(list(xtrain) + list(xvalid))
xtrain_tfv =  tfv.transform(xtrain) 
xvalid_tfv = tfv.transform(xvalid) 


#SVM
Cs=np.logspace(-2,4,num=100)
scores=[]
for C in Cs:
    cls=SVC(C=C, kernel='linear', gamma=1, decision_function_shape='ovo')
    cls.fit(xtrain_tfv ,ytrain)
    scores.append(cls.score(xvalid_tfv,yvalid))
    print(C)
#get the best C of penalty when doing L1 regularizarion   
import matplotlib.pyplot as plt
fig=plt.figure()
ax=fig.add_subplot(1,1,1)
ax.plot(Cs,scores)
ax.set_xlabel(r"C")
ax.set_ylabel(r"score")
ax.set_xscale('log')
ax.set_title("SVM-Stability")    

clf = SVC(C=1, kernel='linear', gamma=1, decision_function_shape='ovo') 
clf.fit(xtrain_tfv, ytrain)
predictions = clf.predict(xvalid_tfv)
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
#The accurary of SVM
accur=(irir+rere)/(irir+rere+reir+irre)
print('The accuracy of SVM classifier is :',+accur)

#The relevant recall
reca=(rere)/(rere+reir)
print('The relevant recall of SVM classifier is :',+reca)
#The irrelevant recall
irca=(irir)/(irre+irir)
print('The irrelevant recall of SVM classifier is :',+irca)
#The relevant precision
repr=(rere)/(rere+irre)
print('The relevant precision of SVM classifier is :',+repr)
#The irrelevant precision
irpr=(irir)/(irir+reir)
print('The irrelevant precision of SVM classifier is :',+irpr)
from sklearn.metrics import roc_curve
fpr, tpr, thresholds = roc_curve(actual, predictions)
print(fpr)
print(tpr)
print(thresholds)
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(actual, predictions)
print("The auc:",+auc)
def class_logloss(actual, predicted, eps=1e-15):
    clip = np.clip(predicted, eps, 1 - eps)
    rows = actual.shape[0]
    vsota = np.sum(actual * np.log(clip))
    return -1.0 / rows * vsota
print ("logloss: %0.3f " % class_logloss(yvalid, predictions))
import matplotlib.pyplot as plt
plt.plot(fpr,tpr)
plt.title("SVM-Stability")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.fill_between(fpr, tpr, where=(tpr>=0), color='Blue', alpha=0.5)
plt.show()