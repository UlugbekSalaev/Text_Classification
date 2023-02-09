import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

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

# Make predictions on the test set
y_pred = model.predict(x_test)

# Convert the predicted probabilities to class labels
# y_pred = [1 if p > 0.5 else 0 for p in y_pred]
y_pred = np.where(y_pred > 0.5, 1, 0)

# Calculate the performance metrics
print(classification_report(y_test, y_pred))

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


# Epoch 1/10
# 12819/12819 - 90s - loss: 0.5369 - accuracy: 0.8139 - val_loss: 19.7072 - val_accuracy: 0.0701 - 90s/epoch - 7ms/step
# Epoch 2/10
# 12819/12819 - 124s - loss: 0.3796 - accuracy: 0.8611 - val_loss: 24.1449 - val_accuracy: 0.0727 - 124s/epoch - 10ms/step
# Epoch 3/10
# 12819/12819 - 114s - loss: 0.3363 - accuracy: 0.8754 - val_loss: 27.7012 - val_accuracy: 0.0734 - 114s/epoch - 9ms/step
# Epoch 4/10
# 12819/12819 - 215s - loss: 0.3056 - accuracy: 0.8872 - val_loss: 31.0089 - val_accuracy: 0.0738 - 215s/epoch - 17ms/step
# Epoch 5/10
# 12819/12819 - 221s - loss: 0.2813 - accuracy: 0.8956 - val_loss: 35.3300 - val_accuracy: 0.0725 - 221s/epoch - 17ms/step
# Epoch 6/10
# 12819/12819 - 228s - loss: 0.2615 - accuracy: 0.9027 - val_loss: 39.5297 - val_accuracy: 0.0738 - 228s/epoch - 18ms/step
# Epoch 7/10
# 12819/12819 - 208s - loss: 0.2435 - accuracy: 0.9093 - val_loss: 40.7730 - val_accuracy: 0.0725 - 208s/epoch - 16ms/step
# Epoch 8/10
# 12819/12819 - 219s - loss: 0.2285 - accuracy: 0.9150 - val_loss: 41.5917 - val_accuracy: 0.0743 - 219s/epoch - 17ms/step
# Epoch 9/10
# 12819/12819 - 232s - loss: 0.2136 - accuracy: 0.9202 - val_loss: 47.3559 - val_accuracy: 0.0739 - 232s/epoch - 18ms/step
# Epoch 10/10
# 12819/12819 - 228s - loss: 0.2017 - accuracy: 0.9245 - val_loss: 46.2253 - val_accuracy: 0.0747 - 228s/epoch - 18ms/step
# 3205/3205 [==============================] - 24s 7ms/step
# C:\Users\E-MaxPCShop\PycharmProjects\Text_Classification\venv\lib\site-packages\sklearn\metrics\_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, msg_start, len(result))
# C:\Users\E-MaxPCShop\PycharmProjects\Text_Classification\venv\lib\site-packages\sklearn\metrics\_classification.py:1344: UndefinedMetricWarning: Precision and F-score are ill-defined and being set to 0.0 in samples with no predicted labels. Use `zero_division` parameter to control this behavior.
#   _warn_prf(average, modifier, msg_start, len(result))
#               precision    recall  f1-score   support
#
#            0       0.81      0.77      0.79      1214
#            1       0.62      0.96      0.76       532
#            2       0.81      0.97      0.88     27282
#            3       0.68      0.90      0.78       789
#            4       0.71      0.82      0.76      2472
#            5       0.67      0.88      0.76     11135
#            6       0.80      0.98      0.88       821
#            7       0.85      0.93      0.89      2555
#            8       0.77      0.95      0.85     29662
#            9       0.97      0.99      0.98       442
#           10       0.91      0.99      0.95      6588
#           11       0.00      0.00      0.00      1053
#           12       0.00      0.00      0.00      2507
#           13       0.00      0.00      0.00     12046
#           14       0.00      0.00      0.00      3452
#
#    micro avg       0.78      0.77      0.77    102550
#    macro avg       0.57      0.68      0.62    102550
# weighted avg       0.64      0.77      0.69    102550
#  samples avg       0.77      0.77      0.77    102550
#
# 3205/3205 - 21s - loss: 9.4363 - accuracy: 0.7693 - 21s/epoch - 7ms/step
# Test loss: 9.436348915100098
# Test accuracy: 0.7692832946777344
# Dataset N =  512750
#
# Process finished with exit code 0