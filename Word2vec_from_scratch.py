#######Note the training time was more than for pre-trained word2Vec embeddings###
####Accuracy still needs to be updated#######################################

import numpy as np
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
import pandas as pd
from keras.callbacks import  EarlyStopping
from nltk.corpus import stopwords
import re,os,time
from keras.layers import Embedding,Flatten
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
stop = (stopwords.words('english'))
from keras.layers import LSTM
import gensim
import nltk

def cleaning_records(df, stop):
    df['description'] = df['description'].astype(str)
    df['description'] = [df['description'][record].lower() for record in range(df.shape[0])]##Lower case###
    df['description'] = [re.sub(r'\n', '',record) for record in df['description']]## Removing \n##
    df_desc = [doc for doc in df.description]
    final = []
    for doc in df_desc:
        temp = doc.split()
        temp = [word for word in temp if word not in stop]
        final.append(' '.join(temp))
    df['description_lower'] = final

    return df

######Loading the data######################################3
path ="../train.csv"

train = pd.read_csv(path)
train['description'] = train['description'].astype(str)

train = cleaning_records(df = train, stop = stop )

categories = train['category'].unique()
    ##Creating dictionaries for category####
dict_categories = {}
for i in range(len(categories)):
    dict_categories[categories[i]] = i
dict_inverse_categories = {i: j for j, i in dict_categories.iteritems()}
train['Labels'] = [dict_categories[category] for category in train['category']]
############################################################

#################################Building word2vec from scratch###########
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

############Tokenizing words from each sentence######################
sentences = []
for sent in train['description_lower'].values:
    sent = sent.decode('utf-8')
    sent_temp = tokenizer.tokenize(sent.strip())
    if len(sent_temp)> 0:
        for elem in sent_temp:
            words = elem.split()
            words = [j for j in words if j not in stop]
            sentences.append(words)

num_features = 300    # Word vector dimensionality
min_word_count = 1   # Minimum word count
num_workers = 4       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model
from gensim.models import word2vec
print "Training model..."
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)


#############Creating a sequence of integers for each train record#################
token = Tokenizer()
token.fit_on_texts(train['description_lower'])
sequences = token.texts_to_sequences(train['description_lower'])

EMBEDDING_DIM = 300
embedding_matrix = np.zeros((len(token.word_index)+ 1, EMBEDDING_DIM))
for word, elem in token.word_index.items():
    try:
        embedding_vector = model[word]
    except KeyError:
        continue
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[elem] = embedding_vector

def return_feature_vector(sequence_to_be_cal,embedding_matrix, seq):
    feature_vector = np.zeros((embedding_matrix.shape[1],))
    for elem in range(len(sequence_to_be_cal)):
        feature_vector = np.add(feature_vector, embedding_matrix[sequence_to_be_cal[elem]])

    if len(sequence_to_be_cal) == 0:
        print seq

    feature_vector = np.divide(feature_vector,len(sequence_to_be_cal))

    return feature_vector

def word2vec_embedding_vectors(sequences, embedding_matrix):
    feature_vector_final = np.zeros((len(sequences),embedding_matrix.shape[1]))
    for seq in  range(len(sequences)):
        feature_vector = return_feature_vector(sequences[seq],embedding_matrix, seq)
        feature_vector_final[seq] = feature_vector

    return feature_vector_final

feature_vector_final = word2vec_embedding_vectors(sequences, embedding_matrix)
######################################################################################

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(penalty= 'l2', multi_class= 'ovr')
start = time.time()
clf.fit(feature_vector_final[:300000], np.array(train.Labels)[:300000])
end = time.time()
pred = clf.predict(feature_vector_final[:300000])
pred_test = clf.predict(feature_vector_final[300000:])

##############################Accuracy###############################################
np.sum(pred == np.array(train.Labels)[:300000])/(1.0 * len(pred)) ###
np.sum(pred_test == np.array(train.Labels)[300000:])/(1.0 * len(pred_test)) #############

##############################End################################################


