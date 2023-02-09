import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# Define the path to the directory containing the subdirectories
path = "C:/Users/ulugbek/Desktop/corpus_utf8/"

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

print("Start training")
# Apply TF-IDF to the texts
vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(texts).toarray()

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Define the model architecture
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation="relu", input_shape=(x.shape[1],)),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(num_classes, activation="softmax")
])

# Compile the model with categorical crossentropy loss and Adam optimizer
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Fit the model on the training data
history = model.fit(x_train, y_train, epochs=10, validation_split=0.2, verbose=2)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
print("Dataset N = ", i)



C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\Scripts\python.exe C:/Users/ulugbek/PycharmProjects/Text_Classification/TF-IDF.py
Start training
Traceback (most recent call last):
  File "C:\Users\ulugbek\PycharmProjects\Text_Classification\TF-IDF.py", line 34, in <module>
    x = vectorizer.fit_transform(texts).toarray()
  File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\scipy\sparse\_compressed.py", line 1051, in toarray
    out = self._process_toarray_args(order, out)
  File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\scipy\sparse\_base.py", line 1298, in _process_toarray_args
    return np.zeros(self.shape, dtype=self.dtype, order=order)
numpy.core._exceptions.MemoryError: Unable to allocate 3.71 TiB for an array with shape (512750, 993175) and data type float64

Process finished with exit code 1
