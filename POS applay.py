import json
import os,ast
import io, re,string
import csv
import random
import numpy as np
import pandas as pd
from pandas import DataFrame
import re
from nltk.tokenize import word_tokenize
from nltk.stem.isri import ISRIStemmer
import itertools
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_validate
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
import nltk
from nltk.tokenize.treebank import TreebankWordDetokenizer
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
#from nltk.tokenize.moses import MosesDetokenizer
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
def untokenize(words):
    """
    Untokenizing a text undoes the tokenizing operation, restoring
    punctuation and spaces to the places that people expect them to be.
    Ideally, `untokenize(tokenize(text))` should be identical to `text`,
    except for line breaks.
    """
    text = ' '.join(words)
    step1 = text.replace("`` ", '"').replace(" ''", '"').replace('. . .',  '...')
    step2 = step1.replace(" ( ", " (").replace(" ) ", ") ")
    step3 = re.sub(r' ([.,:;?!%]+)([ \'"`])', r"\1\2", step2)
    step4 = re.sub(r' ([.,:;?!%]+)$', r"\1", step3)
    step5 = step4.replace(" '", "'").replace(" n't", "n't").replace(
         "can not", "cannot")
    step6 = step5.replace(" ` ", " '")
    return step6.strip()
#ar_sentence = "بسم الله الرحمن الرحيم ، قال الرجل لأخيه لربما تقابلنا مرة أخرى فعسى أن أراك على خير."
def train(classifier, X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=33)
    y_pred = classifier.fit(X_train, y_train).predict(X_test)
    predict = cross_val_predict(classifier, X_test, y_test, cv=10)
    
    scores = cross_val_score(classifier,X_test, y_test, cv=10)
    #print (classifier)
    print (scores)
    
    print ("Accuracy of %s: %0.2f (+/- %0.2f)" % (classifier, scores.mean(), scores.std() *2))
    labels=["academic", "government", "media", "professional", "public"]
    #confusion_mat= confusion_matrix(y_test, y_pred,labels=["academic", "government", "media", "professional", "public"])
    #print(confusion_mat)
    #plt.figure()
    #plot_confusion_matrix(confusion_mat,labels)
    #plt.show()
    return classifier
    

ar_sw_file = open('list.txt', 'r+')
ar_stop_word_list = ar_sw_file.read()
ar_stop_word_list = word_tokenize(ar_stop_word_list)


stemmed_words = []
stemmed_sent = []
#words = word_tokenize(ar_sentence)
st = ISRIStemmer()
#for word in words:
#    if word in ar_stop_word_list:
#        continue
#    stemmed_words.append(st.stem(word))
 #   stemmed_sent = " ".join(stemmed_words)

#print (stemmed_sent)
#print (nltk.pos_tag(stemmed_words))
data_labels=[]
data=[]
data_POS=[]
data_stem=[]
sentence_tags=[]
tags = []
with open('Final_Tweets.csv', 'r',encoding='utf-8') as csvFile:
    reader = csv.reader(csvFile,delimiter=';')
    line_count = 0
    for row in reader:    
        data.append(row[0])
        data_labels.append(row[1])
        words = word_tokenize(row[0])
        #####
        for word in words:
            if word in ar_stop_word_list:
                continue
            stemmed_words.append(st.stem(word))
            stemmed_sent = " ".join(stemmed_words)
            #tags = " ".join(nltk.pos_tag(word)
        data_stem.append(stemmed_sent)
        data_POS.append(nltk.pos_tag(stemmed_words))
        tags=([[tag for word, tag in sent] for sent in data_POS])
        #print(tags)
        
        #sentence_tags.append( TreebankWordDetokenizer().detokenize(tags))
        line_count += 1
    print(f'Processed {line_count} lines.')
    #print(data[1])
    #print(data_POS[1])
    #print(tags[1])
csvFile.close()
ex1=['JJ', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NNP', 'NN']
for x in tags:
    sentence_tags.append( TreebankWordDetokenizer().detokenize(x))
print(sentence_tags[1])    
###print(TreebankWordDetokenizer().detokenize(ex1))
trial4 = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LogisticRegression()),
])
train(trial4, sentence_tags, data_labels)
trial5 = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', RandomForestClassifier()),
])
train(trial5, sentence_tags, data_labels)

trial6 = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB()),
])
train(trial6, sentence_tags, data_labels)
trial7 = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', LinearSVC()),
])
trial8 = Pipeline([
    ('vectorizer',TfidfVectorizer),
    ('classifier', LogisticRegression()),
])
train(trial7, sentence_tags, data_labels)
#train(trial8, sentence_tags, data_labels)
