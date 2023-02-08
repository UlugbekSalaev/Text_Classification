import os
import re
import string
import numpy as np
from gensim.models import KeyedVectors
from gensim.models.word2vec import Text8Corpus

# Preprocess the text in the corpus file
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

path = "C:/Users/E-MaxPCShop/Desktop/corpus_utf8/"

# Get the list of subdirectories (labels)
labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
labels.sort()
label_to_idx = {label: idx for idx, label in enumerate(labels)}

# Load the text files and their corresponding labels
corpus_text = ""
for label in labels:
    for file in os.listdir(os.path.join(path, label)):
        with open(os.path.join(path, label, file), "r", encoding="utf8") as f:
            corpus_text = corpus_text + f.read()


# Preprocess the text
preprocessed_text = preprocess_text(corpus_text)

# Split the preprocessed text into a list of words
corpus_words = preprocessed_text.split()

# Create a Word2Vec model using the Text8Corpus and train it on the corpus
model = KeyedVectors.load_word2vec_format(Text8Corpus(corpus_words), binary=False)

# Set the dimension of the word vectors to 300
model.init_sims(replace=True)
model.vectors = np.concatenate([model.vectors, np.zeros((1, model.vector_size))])

# Save the model to disk
model.save("GloVe_Uzb_Latin_model.h5")
