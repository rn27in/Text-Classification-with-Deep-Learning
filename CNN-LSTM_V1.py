########We only use description of verbatim with title of verbatim not used####
##########We use only 100 dimensions of the Glove Embeddings#############
####################This code is inspired by https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur##

import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils,to_categorical
from keras.layers import Merge
from keras.layers import TimeDistributed, Lambda
from keras.layers import Convolution1D, GlobalMaxPooling1D,SpatialDropout1D
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras import backend as K
from keras.layers.advanced_activations import PReLU
from keras.preprocessing import sequence, text
import nltk
import re,os,time
from nltk.corpus import stopwords
stop = stopwords.words('english')
stop.append('.')
stop = set(stop)

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

MAX_NB_WORDS = 200000

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(train['description'])
sequences = tokenizer.texts_to_sequences(train['description_lower'])

token_inverse = {value:key for key,value in tokenizer.word_index.iteritems()}

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

MAX_SEQUENCE_LENGTH = 40

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,padding= 'post')
labels = to_categorical(np.asarray(train['Labels']))
labels_for_accuracy = np.asarray(train['Labels'])


embeddings_index = {}
f = open(os.path.join('../glove.6B', 'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = 100
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

embedding_layer = Embedding(len(word_index) + 1,
                            EMBEDDING_DIM,
                            weights=[embedding_matrix],
                            input_length=MAX_SEQUENCE_LENGTH,
                            trainable= False)


#model = Sequential()
print('Build model...')

model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                     100,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable= False))

model1.add(TimeDistributed(Dense(100, activation='relu')))
model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(100,)))


###############Trainable Embeddings########################################
model2 = Sequential()
model2.add(Embedding(len(word_index) + 1,
                     100,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=True))

model2.add(TimeDistributed(Dense(100, activation='relu')))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(100,)))
##########################CNN Layer#########################################

nb_filter = 128
filter_length = 5
model3 = Sequential()
model3.add(Embedding(len(word_index) + 1,
                     100,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
model3.add(Convolution1D(filters= nb_filter,
                         kernel_size =filter_length,
                         activation='relu'
                         ))
model3.add(Dropout(0.2))

model3.add(Convolution1D(filters=nb_filter,
                         kernel_size=filter_length,
                         activation='relu'
                         ))

model3.add(GlobalMaxPooling1D())
model3.add(Dropout(0.2))

model3.add(Dense(100))
model3.add(Dropout(0.2))
model3.add(BatchNormalization())

####################################################################################

model4 = Sequential()
model4.add(Embedding(len(word_index) + 1, 100, input_length=40))
model4.add(SpatialDropout1D(0.2))
model4.add(LSTM(100))
model4.add(Dropout(0.2))

######################################################

merged_model = Sequential()
merged_model.add(Merge([model1, model2, model3, model4], mode='concat'))
merged_model.add(BatchNormalization())

merged_model.add(Dense(100))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())
merged_model.add(Dense(100))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(Dense(55))
merged_model.add(Activation('softmax'))

merged_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callback = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=1, mode='auto')

x1 = data[:300000]
y = labels[:300000]
x2 = data[300000:,]
start = time.time()
history = merged_model.fit([x1, x1, x1, x1], y=y, batch_size=384, epochs= 200,
verbose=1, validation_split=0.1, shuffle=True, callbacks= [callback])
end = time.time()

start = time.time()
pred_test = merged_model.predict([x2,x2,x2,x2], verbose = 1)
end = time.time()

############Accuracy###############################################################
pred_test_final = np.argmax(pred_test, axis = 1)
np.sum(pred_test_final == np.array(train.Labels)[300000:])/(1.0 * len(pred_test_final)) ###77.82%

##################################End#############################
