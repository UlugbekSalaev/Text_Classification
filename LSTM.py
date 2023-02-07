import os
import numpy as np
import tensorflow as tf
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from keras.layers import Embedding, LSTM, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split


# Define the path to the directory containing the subdirectories
path = "C:/Users/E-MaxPCShop/Desktop/corpus_utf8"

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

# Convert the labels into one-hot encodings
num_classes = 15
y = tf.keras.utils.to_categorical(y, num_classes)

# Tokenize the texts and pad the sequences to the same length
max_words = 5000
max_length = 250
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x = pad_sequences(sequences, maxlen=max_length)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the LSTM model architecture
model = Sequential()
model.add(Embedding(max_words, 128, input_length=max_length))
model.add(LSTM(64, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(15, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
print("Dataset N = ", i)

# Epoch 1/5
# 12819/12819 [==============================] - 2055s 160ms/step - loss: 0.6667 - accuracy: 0.7847
# Epoch 2/5
# 12819/12819 [==============================] - 1951s 152ms/step - loss: 0.3944 - accuracy: 0.8598
# Epoch 3/5
# 12819/12819 [==============================] - 1948s 152ms/step - loss: 0.3469 - accuracy: 0.8753
# Epoch 4/5
# 12819/12819 [==============================] - 1900s 148ms/step - loss: 0.3187 - accuracy: 0.8844
# Epoch 5/5
# 12819/12819 [==============================] - 1889s 147ms/step - loss: 0.2978 - accuracy: 0.8916
# 3205/3205 - 137s - loss: 0.3681 - accuracy: 0.8706 - 137s/epoch - 43ms/step
# Test loss: 0.3680633008480072
# Test accuracy: 0.8706387281417847
# Dataset N =  512750