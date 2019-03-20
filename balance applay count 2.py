import json
import os,ast
import io, re,string
import csv
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
#import mysql.connector
#from mysql.connector import errorcode, connect
import re
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
import itertools
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#from sklearn import cross_validation
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.pipeline import Pipeline
import string
from nltk.stem import PorterStemmer
from nltk import word_tokenize
from sklearn.feature_selection import chi2
import matplotlib.pyplot as plt
from imblearn.metrics import classification_report_imbalanced
from imblearn.datasets import make_imbalance
from imblearn.under_sampling import NearMiss
from imblearn.pipeline import make_pipeline
from collections import Counter
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline as make_pipeline_imb
from sklearn.ensemble import RandomForestClassifier
def train(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
 
    classifier.fit(X_train, y_train)
    print ("Accuracy: %s" % classifier.score(X_test, y_test))
    return classifier



#########################read csv file#############################
data_labels=[]
data=[]
data_bio=[]
with open('Final_Tweets.csv', 'r',encoding='utf-8') as csvFile:
    reader = csv.reader(csvFile,delimiter=';')
    line_count = 0
    for row in reader:
        
            
        data.append(row[0])
        data_labels.append(row[1])
        #data_bio.append(row[2])    
        line_count += 1
    print(f'Processed {line_count} lines.')

csvFile.close()
#########################


print ("###################logic regression normal with all features#####################")

pipe1 = Pipeline([
    ('vectorizer', CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 2),
    min_df = 2
)),
    ('classifier', LogisticRegression()),
])
train(pipe1, data, data_labels)
print ("###################logic regression OverSample with all features#####################")

pipe3 = make_pipeline_imb(CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 2),
    min_df = 2
),
                         SMOTE(),
                         LogisticRegression())
train(pipe3, data, data_labels)
print ("###################logic regression UnderSample with all features#####################")

pipe4 = make_pipeline_imb(CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 2),
    min_df = 2
),
                         RandomUnderSampler(),
                         LogisticRegression())
train(pipe4, data, data_labels)
#########################


print ("###################RandomForestClassifier normal with all features#####################")

pipe1 = Pipeline([
    ('vectorizer', CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 2),
    min_df = 2
)),
    ('classifier', RandomForestClassifier()),
])
train(pipe1, data, data_labels)
print ("###################RandomForestClassifier OverSample with all features#####################")

pipe3 = make_pipeline_imb(CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 2),
    min_df = 2
),
                         SMOTE(),
                         RandomForestClassifier())
train(pipe3, data, data_labels)
print ("###################RandomForestClassifier UnderSample with all features#####################")

pipe4 = make_pipeline_imb(CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 2),
    min_df = 2
),
                         RandomUnderSampler(),
                         RandomForestClassifier())
train(pipe4, data, data_labels)
#########################


print ("###################MultinomialNB normal with all features#####################")

pipe1 = Pipeline([
    ('vectorizer', CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 2),
    min_df = 2
)),
    ('classifier', LogisticRegression()),
])
train(pipe1, data, data_labels)
print ("###################MultinomialNB OverSample with all features#####################")

pipe3 = make_pipeline_imb(CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 2),
    min_df = 2
),
                         SMOTE(),
                         MultinomialNB())
train(pipe3, data, data_labels)
print ("###################MultinomialNB UnderSample with all features#####################")

pipe4 = make_pipeline_imb(CountVectorizer(),
                         RandomUnderSampler(),
                         MultinomialNB())
train(pipe4, data, data_labels)
#########################


print ("###################LinearSVC normal with all features#####################")

pipe1 = Pipeline([
    ('vectorizer', CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 2),
    min_df = 2
)),
    ('classifier', LinearSVC()),
])
train(pipe1, data, data_labels)
print ("###################LinearSVC OverSample with all features#####################")

pipe3 = make_pipeline_imb(CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 2),
    min_df = 2
),
                         SMOTE(),
                         LinearSVC())
train(pipe3, data, data_labels)
print ("###################logic regression UnderSample with all features#####################")

pipe4 = make_pipeline_imb(CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 2),
    min_df = 2
),
                         RandomUnderSampler(),
                         LinearSVC())
train(pipe4, data, data_labels)
