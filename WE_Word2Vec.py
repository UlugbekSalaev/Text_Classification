import os
import numpy as np
import gensim
from keras.preprocessing.text import Tokenizer
from keras.utils.data_utils import pad_sequences
from keras.layers import Embedding, Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split
import tensorflow as tf

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

# Tokenize the texts and pad the sequences to the same length
max_words = 10000
max_length = 100
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
x = pad_sequences(sequences, maxlen=max_length)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

print("Start training")
model = gensim.models.Word2Vec(texts, vector_size=100, window=5, min_count=5, workers=4) # sg=0 sg=1 sg: The training algorithm, either CBOW(0) or skip gram(1). The default training algorithm is CBOW.

# Get the vocabulary and embedding matrix from the Word2Vec model
# vocabulary = model.wv.vocab
vocabulary = list(model.wv.index_to_key)
embedding_matrix = model.wv.vectors

# Define the model architecture
model = Sequential()
model.add(Embedding(len(vocabulary), 100, weights=[embedding_matrix], input_length=max_length, trainable=False))
model.add(Dense(15, activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=2)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
print("Total dataset item N = ", i)

C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\Scripts\python.exe C:/Users/ulugbek/PycharmProjects/Text_Classification/WE_Word2Vec.py
Start training
2023-02-08 18:55:48.259010: I tensorflow/core/platform/cpu_feature_guard.cc:193] This TensorFlow binary is optimized with oneAPI Deep Neural Network Library (oneDNN) to use the following CPU instructions in performance-critical operations:  AVX AVX2
To enable them in other operations, rebuild TensorFlow with the appropriate compiler flags.
Epoch 1/5
Traceback (most recent call last):
  File "C:\Users\ulugbek\PycharmProjects\Text_Classification\WE_Word2Vec.py", line 62, in <module>
    model.fit(x_train, y_train, epochs=5, validation_split=0.2, verbose=2)
  File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\keras\utils\traceback_utils.py", line 70, in error_handler
    raise e.with_traceback(filtered_tb) from None
  File "C:\Users\ulugbek\AppData\Local\Temp\__autograph_generated_file45_31n0l.py", line 15, in tf__train_function
    retval_ = ag__.converted_call(ag__.ld(step_function), (ag__.ld(self), ag__.ld(iterator)), None, fscope)
ValueError: in user code:

    File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\keras\engine\training.py", line 1249, in train_function  *
        return step_function(self, iterator)
    File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\keras\engine\training.py", line 1233, in step_function  **
        outputs = model.distribute_strategy.run(run_step, args=(data,))
    File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\keras\engine\training.py", line 1222, in run_step  **
        outputs = model.train_step(data)
    File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\keras\engine\training.py", line 1024, in train_step
        loss = self.compute_loss(x, y, y_pred, sample_weight)
    File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\keras\engine\training.py", line 1082, in compute_loss
        return self.compiled_loss(
    File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\keras\engine\compile_utils.py", line 265, in __call__
        loss_value = loss_obj(y_t, y_p, sample_weight=sw)
    File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\keras\losses.py", line 152, in __call__
        losses = call_fn(y_true, y_pred)
    File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\keras\losses.py", line 284, in call  **
        return ag_fn(y_true, y_pred, **self._fn_kwargs)
    File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\keras\losses.py", line 2004, in categorical_crossentropy
        return backend.categorical_crossentropy(
    File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\keras\backend.py", line 5532, in categorical_crossentropy
        target.shape.assert_is_compatible_with(output.shape)

    ValueError: Shapes (32, 15) and (32, 100, 15) are incompatible


Process finished with exit code 1
