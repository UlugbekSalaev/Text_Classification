import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
import tensorflow as tf
import transformers as tfms
from transformers import BertTokenizer, BertForSequenceClassification
from keras.utils import to_categorical
import os

print(torch.cuda.is_available())

# Define the path to the directory containing the subdirectories
path = "C:/Users/ulugbek/Desktop/corpus_utf8"

# Get the list of subdirectories (labels)
labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
labels.sort()
label_to_idx = {label: idx for idx, label in enumerate(labels)}

# Load the text files and their corresponding labels
print("Loading data")
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
# Convert the labels into one-hot encodings
num_classes = 15
y = to_categorical(y, num_classes)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)

# Tokenize the texts using the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
x_train = tokenizer(x_train, padding=True, return_tensors="tf", truncation=True)
x_test = tokenizer(x_test, padding=True, return_tensors="tf", truncation=True)

# Create the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_classes)

# Compile the model
print("Model compiling")
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
print("Train model")
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
print("Total dataset N = ", i)


C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\Scripts\python.exe C:/Users/ulugbek/PycharmProjects/Text_Classification/BERT.py
False
Loading data
Start training
2023-02-08 19:43:20.422712: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
2023-02-08 19:43:20.456868: W tensorflow/tsl/framework/cpu_allocator_impl.cc:82] Allocation of 840089600 exceeds 10% of free system memory.
2023-02-08 19:43:27.171238: W tensorflow/tsl/framework/cpu_allocator_impl.cc:82] Allocation of 840089600 exceeds 10% of free system memory.
2023-02-08 19:43:36.018031: W tensorflow/tsl/framework/cpu_allocator_impl.cc:82] Allocation of 840089600 exceeds 10% of free system memory.
2023-02-08 19:58:49.427307: W tensorflow/tsl/framework/cpu_allocator_impl.cc:82] Allocation of 210022400 exceeds 10% of free system memory.
2023-02-08 19:58:50.238580: W tensorflow/tsl/framework/cpu_allocator_impl.cc:82] Allocation of 210022400 exceeds 10% of free system memory.
Some weights of the model checkpoint at bert-base-uncased were not used when initializing BertForSequenceClassification: ['cls.predictions.transform.LayerNorm.weight', 'cls.predictions.decoder.weight', 'cls.predictions.transform.LayerNorm.bias', 'cls.predictions.bias', 'cls.predictions.transform.dense.bias', 'cls.seq_relationship.bias', 'cls.seq_relationship.weight', 'cls.predictions.transform.dense.weight']
- This IS expected if you are initializing BertForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing BertForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Traceback (most recent call last):
  File "C:\Users\ulugbek\PycharmProjects\Text_Classification\BERT.py", line 51, in <module>
Model compiling
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
  File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\torch\nn\modules\module.py", line 1269, in __getattr__
    raise AttributeError("'{}' object has no attribute '{}'".format(
AttributeError: 'BertForSequenceClassification' object has no attribute 'compile'

Process finished with exit code 1
