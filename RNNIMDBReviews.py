import numpy as np
import pandas as pd
import re
import nltk
import os
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

data_source_url = "https://raw.githubusercontent.com/javaidnabi31/Word-Embeddding-Sentiment-Classification/master/movie_data.csv"
df = pd.read_csv(data_source_url)
#print(df.head(3))

x_train = df.loc[:24999, 'review'].values
y_train = df.loc[:24999, 'sentiment'].values
x_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values

X = np.concatenate((x_train, x_test), axis=0)
y = np.concatenate((y_train, y_test), axis=0)

#print(X.shape)
#print(y.shape)

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


tokenizer_obj = Tokenizer()
total_reviews = x_train + x_test
tokenizer_obj.fit_on_texts(total_reviews)

#pad_sequences
max_length = 100
#max([len(s.split()) for s in total_reviews])

vocab_size = len(tokenizer_obj.word_index) + 1

x_train_tokens = tokenizer_obj.texts_to_sequences(x_train)
x_test_tokens = tokenizer_obj.texts_to_sequences(x_test)

x_train_pad = pad_sequences(x_train_tokens, maxlen=max_length, padding = 'post')
x_test_pad = pad_sequences(x_test_tokens, maxlen=max_length, padding = 'post')

#print(vocab_size)
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
start_time = time.time()


EMBEDDING_DIM = 100

model = Sequential()
model.add(Embedding(vocab_size, EMBEDDING_DIM, input_length=max_length))
model.add(LSTM(100))
model.add(Dense(1, activation = 'sigmoid'))
model.compile(loss = 'binary_crossentropy', optimizer= 'adam', metrics = ['accuracy'])
model.fit(x_train_pad, y_train, batch_size = 128, epochs=25, validation_data=(x_test_pad, y_test), verbose = 2)

#print('Summar of built model...')
#print(model.summary())

from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
#print('train:')
model.fit(x_train_pad, y_train, batch_size = 128, epochs=25, validation_data=(x_test_pad, y_test), verbose = 2)

y_pred = model.predict(x_test_pad)

cm = confusion_matrix(y_test, y_pred.round())

print('Testing...')
score, acc = model.evaluate(x_test_pad, y_test, batch_size=128)
print(cm)

print('Accuracy Score:', accuracy_score(y_test, y_pred.round(), normalize = False))
print('report:')
print(classification_report(y_test, y_pred.round()))
#print('Test score:', score)
#print('Test accuracy:', acc)
#print("Accuracy: {0:.2%}".format(acc))
end_time = time.time()
print(end_time - start_time)
