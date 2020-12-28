

import numpy as np
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, GRU, Embedding

Dataset = ["Mohith is a Data Science Enthusiast and He Likes to Code"]
Dataset = [i.lower() for i in Dataset]
chars = sorted(list(set(Dataset[0])))
mapping = dict((c, i) for i, c in enumerate(chars))

def create_seq(text):
    length = 6
    sequences = list()
    for i in range(length, len(text)):
        seq = text[i-length:i+1]
        sequences.append(seq)
    return sequences

def encode_seq(seq):
    sequences = list()
    for line in seq:
        encoded_seq = [mapping[char] for char in line]
        sequences.append(encoded_seq)
    return sequences

def split_x_y():
    sequences = encode_seq(create_seq(Dataset[0]))
    sequences = np.array(sequences)
    x, y =sequences[:,:-1], sequences[:,-1]
    y = to_categorical(y, num_classes=len(mapping))
    return x,y

def build_model():
    x, y = split_x_y()
    model = Sequential()
    model.add(Embedding(len(mapping), 50, input_length=6, trainable=True))
    model.add(GRU(150, recurrent_dropout=0.1, dropout=0.1))
    model.add(Dense(len(mapping), activation='softmax'))
    model.compile(loss='categorical_crossentropy', metrics=['acc'], optimizer='adam')
    model.fit(x, y, epochs=100)
    return model


def predict(model, mapping, seq_length, seed_text, n_chars):
    in_text = seed_text
    for _ in range(n_chars):
        encoded = [mapping[char] for char in in_text]
        encoded = pad_sequences([encoded], maxlen=seq_length, truncating='pre')
        yhat = model.predict_classes(encoded, verbose=0)
        out_char = ''
        for char, index in mapping.items():
            if index == yhat:
                out_char = char
                break
        in_text += char
    print(in_text)
    return

predict(build_model(),mapping,6,"mohith",50)

