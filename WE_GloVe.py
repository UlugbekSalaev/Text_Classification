# to use Glove model, we need GloVe pre-trained model for Uzbek (latin)
# or we can do training a model for GloVe on Uzbek (latin)
import numpy as np
import os
import pandas as pd
import keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GlobalMaxPooling1D, Dropout
from sklearn.model_selection import train_test_split

# Define the path to the directory containing the subdirectories
path = "C:/Users/E-MaxPCShop/Desktop/corpus_utf8/"

# Get the list of subdirectories (labels)
labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
labels.sort()
label_to_idx = {label: idx for idx, label in enumerate(labels)}

# Load the text files and their corresponding labels
texts = []
y = []
i = 0
for label in labels:
    for file in os.listdir(os.path.join(path, label)):
        with open(os.path.join(path, label, file), "r", encoding="utf8") as f:
            i = i + 1
            texts.append(f.read())
            y.append(label_to_idx[label])

print("Start training")

# Tokenize the texts
tokenizer = Tokenizer()
tokenizer.fit_on_texts(texts)
text_sequences = tokenizer.texts_to_sequences(texts)

# Pad sequences to the same length
max_length = max([len(s) for s in text_sequences])
padded_sequences = pad_sequences(text_sequences, maxlen=max_length)

# Create the GloVe embedding matrix
embeddings_index = {}
with open("glove.6B.300d.txt", encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

embedding_matrix = np.zeros((len(tokenizer.word_index) + 1, 300))
for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(padded_sequences, y, test_size=0.2, random_state=42)

# Define the model architecture
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 300, weights=[embedding_matrix], input_length=max_length, trainable=False))
model.add(LSTM(100, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))

# Compile the model with binary crossentropy loss and Adam optimizer
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Fit the model on the training data
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test), batch_size=32, verbose=2)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
print("Dataset N = ",i)
