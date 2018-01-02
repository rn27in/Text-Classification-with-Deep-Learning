###https://www.linkedin.com/pulse/duplicate-quora-question-abhishek-thakur####################
##########This uses the same structure as V1################
#######We use 300 dimensions for Glove Embeddings#############
######We use both description and title of the verbatim##############
#########Training time is roughly 11 hours and testing time is 22 minutes#############
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

def cleaning_records_title(df, stop):
    df['title'] = df['title'].astype(str)
    df['title'] = [df['title'][record].lower() for record in range(df.shape[0])]##Lower case###
    df['title'] = [re.sub(r'\n', '',record) for record in df['title']]## Removing \n##
    df_desc = [doc for doc in df.title]
    final = []
    for doc in df_desc:
        temp = doc.split()
        temp = [word for word in temp if word not in stop]
        final.append(' '.join(temp))
    df['title_lower'] = final

    return df

path ="../train.csv"

train = pd.read_csv(path)
train['description'] = train['description'].astype(str)

train = cleaning_records(df = train, stop = stop )
train = cleaning_records_title(df = train, stop = stop )

categories = train['category'].unique()
    ##Creating dictionaries for category####
dict_categories = {}
for i in range(len(categories)):
    dict_categories[categories[i]] = i
dict_inverse_categories = {i: j for j, i in dict_categories.iteritems()}
train['Labels'] = [dict_categories[category] for category in train['category']]

MAX_NB_WORDS = 200000

tokenizer = Tokenizer(num_words= MAX_NB_WORDS)
tokenizer.fit_on_texts(list(train['description_lower'].values) + list(train['title_lower'].values))
sequences_desc = tokenizer.texts_to_sequences(train['description_lower'])
sequences_title = tokenizer.texts_to_sequences(train['title_lower'])

token_inverse = {value:key for key,value in tokenizer.word_index.iteritems()}

word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

MAX_SEQUENCE_LENGTH = 40

data_desc = pad_sequences(sequences_desc, maxlen=MAX_SEQUENCE_LENGTH,padding= 'post')
data_title = pad_sequences(sequences_title, maxlen=MAX_SEQUENCE_LENGTH,padding= 'post')
labels = to_categorical(np.asarray(train['Labels']))
labels_for_accuracy = np.asarray(train['Labels'])


embeddings_index = {}
f = open(os.path.join('../glove.6B', 'glove.6B.300d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

EMBEDDING_DIM = 300
embedding_matrix = np.zeros((len(word_index) + 1, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector


#model = Sequential()
print('Build model...')

model1 = Sequential()
model1.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable= False))

model1.add(TimeDistributed(Dense(300, activation='relu')))
model1.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))


###############Trainable Embeddings########################################
model2 = Sequential()
model2.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=True))

model2.add(TimeDistributed(Dense(300, activation='relu')))
model2.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))
##########################CNN Layer#########################################

nb_filter = 128
filter_length = 5
model3 = Sequential()
model3.add(Embedding(len(word_index) + 1,
                     300,
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
model4.add(Embedding(len(word_index) + 1, 300, input_length=40))
model4.add(SpatialDropout1D(0.2))
model4.add(LSTM(300))
model4.add(Dropout(0.2))

######################################################
############We create same thing for title now############
model1a = Sequential()
model1a.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable= False))

model1a.add(TimeDistributed(Dense(300, activation='relu')))
model1a.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))


###############Trainable Embeddings########################################
model2a = Sequential()
model2a.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=True))

model2a.add(TimeDistributed(Dense(300, activation='relu')))
model2a.add(Lambda(lambda x: K.sum(x, axis=1), output_shape=(300,)))

##################################################################################


nb_filter = 128
filter_length = 5
model3a = Sequential()
model3a.add(Embedding(len(word_index) + 1,
                     300,
                     weights=[embedding_matrix],
                     input_length=40,
                     trainable=False))
model3a.add(Convolution1D(filters= nb_filter,
                         kernel_size =filter_length,
                         activation='relu'
                         ))
model3a.add(Dropout(0.2))

model3a.add(Convolution1D(filters=nb_filter,
                         kernel_size=filter_length,
                         activation='relu'
                         ))

model3a.add(GlobalMaxPooling1D())
model3a.add(Dropout(0.2))

model3a.add(Dense(100))
model3a.add(Dropout(0.2))
model3a.add(BatchNormalization())

####################################################################################

model4a = Sequential()
model4a.add(Embedding(len(word_index) + 1, 300, input_length=40))
model4a.add(SpatialDropout1D(0.2))
model4a.add(LSTM(300))
model4a.add(Dropout(0.2))

###################################################################################
merged_model = Sequential()
merged_model.add(Merge([model1, model2, model3, model4, model1a, model2a, model3a, model4a], mode='concat'))
merged_model.add(BatchNormalization())

##############################################################
merged_model.add(Dense(100))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(BatchNormalization())
merged_model.add(Dense(100))
merged_model.add(PReLU())
merged_model.add(Dropout(0.2))
merged_model.add(Dense(55))
merged_model.add(Activation('softmax'))
#########################


merged_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

callback = EarlyStopping(monitor='val_loss',
                              min_delta=0,
                              patience=5,
                              verbose=1, mode='auto')

# x1 = data_desc[:300000]
# y = labels[:300000]
# x2 = data_title[:300000]
# x3 = data_desc[300000:312000,]
# x4 = data_title[300000:312000,]

start = time.time()
history = merged_model.fit([data_desc[:300000], data_desc[:300000], data_desc[:300000], data_desc[:300000],data_title[:300000],data_title[:300000],data_title[:300000],data_title[:300000]], y= labels[:300000], batch_size=384, epochs= 200,
verbose=1, validation_split=0.1, shuffle=True, callbacks= [callback])
end = time.time()

start = time.time()
pred_test = merged_model.predict([data_desc[300000:], data_desc[300000:], data_desc[300000:], data_desc[300000:],data_title[300000:],data_title[300000:],data_title[300000:],data_title[300000:]], verbose = 1)
end = time.time()

pred_test_final = np.argmax(pred_test, axis = 1)
np.sum(pred_test_final == np.array(train.Labels)[300000:])/(1.0 * len(pred_test_final)) ###77.82%

from sklearn.externals import joblib
joblib.dump(pred_test_final, '../preds_83.73_percent')
joblib.dump(history.history,'../history_83.73_percent')

from keras.models import model_from_json
#Saving Keras models to disk##########
##Assume the model is called fc_model######
model_json = merged_model.to_json()
with open("../model_4_models_merged_incudes_title.json", "w") as json_file:
    json_file.write(model_json)

merged_model.save_weights("../model_4_models_merged_includes_title.h5")
print("Saved model to disk")
