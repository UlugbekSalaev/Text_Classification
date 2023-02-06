import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
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
max_words = 10000
max_length = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x = pad_sequences(sequences, maxlen=max_length)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(max_words, 16, input_length=max_length),
    tf.keras.layers.LSTM(100, return_sequences=False),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

# Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit the model on the training data
history = model.fit(x, y, epochs=10, validation_split=0.2, verbose=2)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
print("Dataset N = ", i)

# Epoch 1/10
# 12819/12819 - 508s - loss: 0.6154 - accuracy: 0.7917 - val_loss: 13.3289 - val_accuracy: 0.0723 - 508s/epoch - 40ms/step
# Epoch 2/10
# 12819/12819 - 516s - loss: 0.3960 - accuracy: 0.8547 - val_loss: 15.8498 - val_accuracy: 0.0722 - 516s/epoch - 40ms/step
# Epoch 3/10
# 12819/12819 - 535s - loss: 0.3434 - accuracy: 0.8730 - val_loss: 17.9232 - val_accuracy: 0.0746 - 535s/epoch - 42ms/step
# Epoch 4/10
# 12819/12819 - 549s - loss: 0.3118 - accuracy: 0.8843 - val_loss: 18.4807 - val_accuracy: 0.0729 - 549s/epoch - 43ms/step
# Epoch 5/10
# 12819/12819 - 571s - loss: 0.2876 - accuracy: 0.8924 - val_loss: 18.8936 - val_accuracy: 0.0710 - 571s/epoch - 45ms/step
# Epoch 6/10
# 12819/12819 - 552s - loss: 0.2665 - accuracy: 0.9005 - val_loss: 19.4336 - val_accuracy: 0.0743 - 552s/epoch - 43ms/step
# Epoch 7/10
# 12819/12819 - 542s - loss: 0.2474 - accuracy: 0.9077 - val_loss: 20.4648 - val_accuracy: 0.0728 - 542s/epoch - 42ms/step
# Epoch 8/10
# 12819/12819 - 567s - loss: 0.2294 - accuracy: 0.9145 - val_loss: 20.7086 - val_accuracy: 0.0732 - 567s/epoch - 44ms/step
# Epoch 9/10
# 12819/12819 - 892s - loss: 0.2124 - accuracy: 0.9214 - val_loss: 21.5485 - val_accuracy: 0.0741 - 892s/epoch - 70ms/step
# Epoch 10/10
# 12819/12819 - 1226s - loss: 0.1959 - accuracy: 0.9276 - val_loss: 22.0861 - val_accuracy: 0.0739 - 1226s/epoch - 96ms/step
# 3205/3205 - 56s - loss: 4.5747 - accuracy: 0.7701 - 56s/epoch - 18ms/step
# Test loss: 4.574697494506836
# Test accuracy: 0.7700634002685547
# Dataset N =  512750