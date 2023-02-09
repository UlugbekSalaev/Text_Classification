import os

path = "C:/Users/E-MaxPCShop/Desktop/corpus_utf8/"

labels = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
labels.sort()
label_to_idx = {label: idx for idx, label in enumerate(labels)}

# Load the text files and their corresponding labels
texts = []
total = 0
for label in labels:
    i = 0
    total_word = 0
    total_char = 0
    for file in os.listdir(os.path.join(path, label)):
        with open(os.path.join(path, label, file), "r", encoding="utf8") as f:
            i = i + 1
            text = f.read().strip()
            words = len(text.split())
            chars = len(text)
            total_word = total_word + words
            total_char = total_char + chars
    total = total + i
    print(label, '\t', i, '\t', total_word, '\t', round(total_word/i), '\t', total_char, '\t', round(total_char/i))

print("Total=", total)
