import json
import os,ast
import io, re,string
import csv
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
import itertools
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectPercentile
import csv
from sklearn.feature_selection import SelectFromModel
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
###################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
#######################################
def featureExtraction1(data):
    vectorizer = TfidfVectorizer(min_df=10, max_df=0.75, ngram_range=(1,3))
    tfidf_data = vectorizer.fit_transform(data)
    return tfidf_data

#######################
def featureExtraction2(data):
    vectorizer =  CountVectorizer(
    analyzer = 'word',
    lowercase = False,
    ngram_range = (1, 3),
    min_df = 2
)
    count_data = vectorizer.fit_transform(data)
    return count_data


####################
def learning(clf, X, Y):
   
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.25, random_state=33)
    
    classifer = clf()
    classifer.fit(X_train, Y_train)
    
    #predict = cross_val_predict(classifer, X_test, Y_test, cv=10)
    
    #scores = cross_val_score(classifer,X_test, Y_test, cv=10)
    #y_pred = classifer.fit(X_train, Y_train).predict(X_test)
    #print (scores)
    
    #print ("Accuracy of %s: %0.2f (+/- %0.2f)" % (classifer, scores.mean(), scores.std() *2))
    print(classifer)
    print ("###############Accuracy: %s" % classifer.score(X_test, Y_test))
    #print (classification_report(Y_test, predict))
    labels=["academic", "government", "media", "professional", "public"]
    #confusion_mat= confusion_matrix(Y_test, y_pred,labels=["academic", "government", "media", "professional", "public"])
    #print(confusion_mat)
    #plt.figure()
    #plot_confusion_matrix(confusion_mat,labels)
    #plt.show()
    select = SelectPercentile(percentile=60)
    select.fit(X_train, Y_train)
    X_train_selected = select.transform(X_train)
    X_test_selected = select.transform(X_test)
    classifer.fit(X_train_selected, Y_train)
    print ("###############Accuracy of selectePercentile: %s" % classifer.score(X_test_selected, Y_test))
    select2 = SelectFromModel(RandomForestClassifier(n_estimators=100, random_state=42), threshold='median')
    #select2 = SelectFromModel(LogisticRegression(), threshold='median')
    select2.fit(X_train, Y_train)
    X_train_selected = select2.transform(X_train)
    X_test_selected = select2.transform(X_test)
    classifer.fit(X_train_selected, Y_train)
    print ("###############Accuracy of selecteModel: %s" % classifer.score(X_test_selected, Y_test))
    
#########################read csv file#############################
data_labels=[]
data=[]
bio=[]
data_bio=[]
with open('Final_Tweets.csv', 'r',encoding='utf-8') as csvFile:
    reader = csv.reader(csvFile,delimiter=';')
    line_count = 0
    for row in reader:    
        data.append(row[0])
        data_labels.append(row[1])
        #bio.append(row[2])
        data_bio.append(row[0]+row[2])
        #data_bio.append(row[2])
        line_count += 1
    print(f'Processed {line_count} lines.')

csvFile.close()
count_data = featureExtraction2(data_bio)
clf = LogisticRegression
learning(clf,count_data,data_labels)
clf = RandomForestClassifier
learning(clf,count_data,data_labels)
clf=MultinomialNB
learning(clf,count_data,data_labels)
clf = LinearSVC
learning(clf,count_data,data_labels)


