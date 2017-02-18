
# coding: utf-8

# In[2]:

from __future__ import division, print_function, absolute_import
import tflearn
from tflearn.data_utils import to_categorical, pad_sequences


# In[1]:

import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import *
import pickle
import sklearn
from sklearn.model_selection import train_test_split
# nltk.download()
#%pylab inline
# from sklearn.metrics import accuracy_score


# In[4]:

def print_head(lines, n) :
    print (len(lines))
    for i in range(n) :
        print (lines[i])


# In[7]:

def build_dictionary() :
    max_length_1 = 0
    
    sentiment_words_set = set()
    cachedStopWords = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    
    with open('products_sentiment_train.tsv', 'r') as f0:
        for line in f0:
            # print(line)
            words = re.findall("\w{2,}", line.lower() )
            if len(words) > max_length_1 :
                max_length_1 = len(words)
            # print(words)
            for w in words:
                if w not in cachedStopWords :
                    if not w.isdigit()  :
                        sentiment_words_set.add(stemmer.stem(w))
                
        
    max_length_2 = 0
    with open('products_sentiment_test.tsv', 'r') as f1:
        first = True
        for line in f1:
            if first :
                first = False
                continue
            #print (line)
            words = re.findall('\w{2,}', line[2:].lower())
            if len(words) > max_length_2 :
                max_length_2 = len(words)
            #print (words)
            for w in words:
                if w not in cachedStopWords :
                    if not w.isdigit()  :
                        # print (w)
                        sentiment_words_set.add(stemmer.stem(w))
        
    sentiment_dictionary = { key: i for i, key in enumerate(sentiment_words_set, start=1) }
    
    with open('kaggle_dictionary.pkl', 'wb') as f3:
            pickle.dump(sentiment_dictionary, f3 )
      
    print ("words in dictionary:", len(sentiment_dictionary), "   max number words in topic:", 
           (max_length_1 if max_length_1 > max_length_2 else max_length_2))
    
    return sentiment_dictionary


# In[8]:

def load_train_data(word_dict) :
    cachedStopWords = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    train_data = list()
    train_labels = list()
    with open('products_sentiment_train.tsv', 'r') as f2:
        for line in f2:
            train_labels.append(int(line[-2:]) )
            words = re.findall('\w{2,}', line.lower())
            train_line = list()
            # print (words)
            for w in words:
                if w not in cachedStopWords :
                    if not w.isdigit()  :
                        train_line.append(word_dict.get(stemmer.stem(w)))
            
            train_data.append(train_line)
            
    return train_data, train_labels


# In[9]:

def load_test_data(word_dict) :
    cachedStopWords = stopwords.words("english")
    stemmer = SnowballStemmer("english")
    test_data = list()
    first = True
    with open('products_sentiment_test.tsv', 'r') as f3:
         for line in f3:
            if first :
                first = False
                continue
            words = re.findall('\w{2,}', line.lower())
            test_line = list()
            # print (words)
            for w in words:
                if w not in cachedStopWords :
                    if not w.isdigit()  :
                        test_line.append(word_dict.get(stemmer.stem(w)))
                
            test_data.append(test_line)
            
    return test_data


# In[10]:

mydict = build_dictionary()


# In[ ]:

# model layers
net = tflearn.input_data([None, 100])
net = tflearn.embedding(net, input_dim=3200, output_dim=128)
net = tflearn.lstm(net, 128, dropout=0.5, return_seq=True)
net = tflearn.lstm(net, 128, dropout=0.5)
net = tflearn.fully_connected(net, 128)
net = tflearn.fully_connected(net, 2, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy')
model = tflearn.DNN(net, tensorboard_verbose=0)


# In[ ]:

X, Y = load_train_data(mydict)


# In[ ]:

X_train, X_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.05, random_state=7 )


# In[ ]:

# Data preprocessing
# Sequence padding
trainX = pad_sequences(X_train, maxlen=100, value=0.)
validX = pad_sequences(X_valid, maxlen=100, value=0.)
# Converting labels to binary vectors
trainY = to_categorical(y_train, nb_classes=2)
validY = to_categorical(y_valid, nb_classes=2)


# In[ ]:

model.fit(trainX, trainY, validation_set=(validX, validY), show_metric=True, n_epoch=10, batch_size=50)


# In[ ]:

testX = pad_sequences(load_test_data(mydict), maxlen=100, value=0.)


# In[ ]:

predict = model.predict(testX)


# In[ ]:

kaggle_predict = [ int(round(pr[1])) for pr in predict ]


# In[ ]:

print (kaggle_predict[:5])


# In[ ]:

#print_head(kaggle_predict, 30)


# In[ ]:

with open('sokolovdp_full_summission.txt', 'w') as f5:
    f5.write("Id,y\n")
    for i, pr in enumerate(kaggle_predict) :
        f5.write("%d,%d\n"% (i, pr) )


# In[ ]:



