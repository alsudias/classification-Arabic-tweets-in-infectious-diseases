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
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
import itertools
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn import cross_validation
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from nltk import agreement
###################################    
#########################read csv file#############################
data_labels_coder1=[]
data_labels_coder2=[]
data=[]

with open('Final_Tweets_coders.csv', 'r',encoding='utf-8') as csvFile:
    reader = csv.reader(csvFile,delimiter=';')
    line_count = 0
    for row in reader:    
        data.append(row[0])
        data_labels_coder1.append(row[1])
        data_labels_coder2.append(row[4])
        line_count += 1
    print(f'Processed {line_count} lines.')

csvFile.close()
kappa = cohen_kappa_score(data_labels_coder1, data_labels_coder2)
print(kappa)
taskdata=[[0,str(i),str(data_labels_coder1[i])] for i in range(0,len(data_labels_coder1))]+[[1,str(i),str(data_labels_coder2[i])] for i in range(0,len(data_labels_coder2))]
ratingtask = agreement.AnnotationTask(data=taskdata)
print("kappa " +str(ratingtask.kappa()))
print("fleiss " + str(ratingtask.multi_kappa()))
print("alpha " +str(ratingtask.alpha()))
print("scotts " + str(ratingtask.pi()))

