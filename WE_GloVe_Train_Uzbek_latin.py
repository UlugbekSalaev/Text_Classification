import os
import numpy as np
import gensim.downloader as api

from gensim.scripts.glove2word2vec import glove2word2vec
from gensim.models import KeyedVectors

# Define the path to your text corpus
corpus_path = "C:/Users/E-MaxPCShop/Desktop/corpus_utf8/Avto/1.txt"

# Define the path to save the resulting word embeddings
embeddings_path = "embeddings.txt"

# Train the GloVe model on your text corpus
glove_model = api.load("glove-wiki-gigaword-300")
glove2word2vec(glove_model.word_vec, embeddings_path)

# Load the trained word embeddings
word_vectors = KeyedVectors.load_word2vec_format(embeddings_path, binary=False)

# Get the word vector for a given word
word_vector = word_vectors["word"]