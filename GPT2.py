import os
import tensorflow as tf
import torch
import transformers as tfm
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
            texts.append(f.read())
            i = i + 1
            y.append(label_to_idx[label])

print("Start training")
# Convert the labels into one-hot encodings
num_classes = 15
y = tf.keras.utils.to_categorical(y, num_classes)

# Split the data into training and test sets
x_train, x_test, y_train, y_test = train_test_split(texts, y, test_size=0.2, random_state=42)

# Load the pre-trained GPT-2 model
model = tfm.GPT2Model.from_pretrained("gpt2")
#Freeze the pre=trained weight
for param in model.parametres():
    param.requires_grad = False

#Add a linear layer for classification
model.classifier = torch.nn.Linear(model.config.hidden_size, num_classes)

# Add a custom layer for classification
# classification_layer = tf.keras.layers.Dense(15, activation='softmax')
# model.layers.append(classification_layer)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model on the new classification task
model.fit(x_train, y_train, epochs=5, batch_size=32)

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=2)
print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)
print("Total dataset N = ", i)


C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\Scripts\python.exe C:/Users/ulugbek/PycharmProjects/Text_Classification/GPT2.py
Start training
Traceback (most recent call last):
  File "C:\Users\ulugbek\PycharmProjects\Text_Classification\GPT2.py", line 37, in <module>
    for param in model.parametres():
  File "C:\Users\ulugbek\PycharmProjects\Text_Classification\venv\lib\site-packages\torch\nn\modules\module.py", line 1269, in __getattr__
    raise AttributeError("'{}' object has no attribute '{}'".format(
AttributeError: 'GPT2Model' object has no attribute 'parametres'

Process finished with exit code 1
