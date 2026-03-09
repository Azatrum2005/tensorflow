file = open('Royal_data.txt', 'r')
royal_data = file.readlines()
print(royal_data)
file.close()

for i in range(len(royal_data)):
    royal_data[i] = royal_data[i].lower().replace('\n', '')
print(royal_data)

stopwords = ['the', 'is', 'will', 'be', 'a', 'only', 'can', 'their', 'now', 'and', 'at', 'it']
filtered_data = []
for sent in royal_data:
    temp = []
    for word in sent.split():
        if word not in stopwords:
            temp.append(word)
    filtered_data.append(temp)
print(filtered_data)

bigrams = []
for words_list in filtered_data:
    for i in range(len(words_list) - 1):
        for j in range(i+1, len(words_list)):
            bigrams.append([words_list[i], words_list[j]])
            bigrams.append([words_list[j], words_list[i]])
print(bigrams)

all_words = []
for bi in bigrams:
    all_words.extend(bi)
all_words = list(set(all_words))
all_words.sort()
print(all_words)
print("Total number of unique words are:", len(all_words))

words_dict = {}
counter = 0
for word in all_words:
    words_dict[word] = counter
    counter += 1
print(words_dict)

import numpy as np
onehot_data = np.zeros((len(all_words), len(all_words)))
for i in range(len(all_words)):
    onehot_data[i][i] = 1
onehot_dict = {}
counter = 0
for word in all_words:
    onehot_dict[word] = onehot_data[counter]
    counter += 1
for word in onehot_dict:
    print(word, ":", onehot_dict[word])

X = []
Y = []
for bi in bigrams:
    X.append(onehot_dict[bi[0]])
    Y.append(onehot_dict[bi[1]])
X = np.array(X)
Y = np.array(Y)

import tensorflow as tf
layers=tf.keras.layers
Sequential=tf.keras.models.Sequential
Dense=layers.Dense
embed_size = 2
model = Sequential([
    Dense(embed_size, activation='linear'),
    Dense(Y.shape[1], activation = 'softmax')
])
model.compile(loss = 'categorical_crossentropy', optimizer = 'adam')
model.fit(X, Y, epochs = 1000, batch_size = 256, verbose = False)

weights = model.get_weights()[0]
print(weights)
word_embeddings = {}
for word in all_words:
    word_embeddings[word] = weights[words_dict[word]]
print(word_embeddings)

import matplotlib.pyplot as plt
# plt.figure(figsize = (10, 10))
for word in list(words_dict.keys()):
    coord = word_embeddings.get(word)
    plt.scatter(coord[0], coord[1])
    plt.annotate(word, (coord[0], coord[1]))
plt.show()
# plt.savefig('img.jpg')