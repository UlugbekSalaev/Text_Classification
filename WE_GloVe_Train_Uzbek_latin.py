import numpy as np
import os
import gensim
from gensim.scripts.glove2word2vec import glove2word2vec

# Define the path to the directory containing the subdirectories
path = "C:/Users/E-MaxPCShop/Desktop/corpus_utf8/"

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

print("Start training")
# Train the model using the corpus
model = gensim.models.Word2Vec(sentences=texts, vector_size=300, window=5, min_count=5, workers=4)

# Save the model
model.save("glove_model")

print("Convrt model to W2V")
# Convert the model to word2vec format
glove2word2vec(glove_input_file="glove_model", word2vec_output_file="glove_word2vec_format", encodings = "utf8")

# Load the converted model
embedding_matrix = gensim.models.KeyedVectors.load_word2vec_format("glove_word2vec_format", binary=False)

# Save the embedding matrix
np.save("glove_embedding_matrix.npy", embedding_matrix.vectors)