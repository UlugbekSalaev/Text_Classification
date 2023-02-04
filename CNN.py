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
    tf.keras.layers.Conv1D(128, 5, activation="relu"),
    tf.keras.layers.GlobalMaxPool1D(),
    tf.keras.layers.Dense(64, activation="relu"),
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

# ---------------
# Epoch 1/10
# 12819/12819 - 63s - loss: 0.5307 - accuracy: 0.8159 - val_loss: 17.4542 - val_accuracy: 0.0734 - 63s/epoch - 5ms/step
# Epoch 2/10
# 12819/12819 - 69s - loss: 0.3795 - accuracy: 0.8606 - val_loss: 23.4386 - val_accuracy: 0.0751 - 69s/epoch - 5ms/step
# Epoch 3/10
# 12819/12819 - 67s - loss: 0.3365 - accuracy: 0.8761 - val_loss: 23.9441 - val_accuracy: 0.0741 - 67s/epoch - 5ms/step
# Epoch 4/10
# 12819/12819 - 67s - loss: 0.3059 - accuracy: 0.8865 - val_loss: 26.7154 - val_accuracy: 0.0733 - 67s/epoch - 5ms/step
# Epoch 5/10
# 12819/12819 - 67s - loss: 0.2821 - accuracy: 0.8948 - val_loss: 32.0303 - val_accuracy: 0.0717 - 67s/epoch - 5ms/step
# Epoch 6/10
# 12819/12819 - 67s - loss: 0.2619 - accuracy: 0.9026 - val_loss: 30.0574 - val_accuracy: 0.0738 - 67s/epoch - 5ms/step
# Epoch 7/10
# 12819/12819 - 67s - loss: 0.2437 - accuracy: 0.9088 - val_loss: 33.4628 - val_accuracy: 0.0738 - 67s/epoch - 5ms/step
# Epoch 8/10
# 12819/12819 - 69s - loss: 0.2281 - accuracy: 0.9143 - val_loss: 34.6717 - val_accuracy: 0.0740 - 69s/epoch - 5ms/step
# Epoch 9/10
# 12819/12819 - 63s - loss: 0.2129 - accuracy: 0.9205 - val_loss: 36.7898 - val_accuracy: 0.0740 - 63s/epoch - 5ms/step
# Epoch 10/10
# 12819/12819 - 63s - loss: 0.2002 - accuracy: 0.9252 - val_loss: 40.2890 - val_accuracy: 0.0741 - 63s/epoch - 5ms/step
# 3205/3205 - 8s - loss: 8.2088 - accuracy: 0.7684 - 8s/epoch - 2ms/step
# Test loss: 8.208784103393555
# Test accuracy: 0.7684251666069031
# Dataset N =  512750