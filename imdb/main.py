"""
Tensorflow NNs to recognize good or bad IMDB reviews
"""

import tensorflow as tf
from tensorflow import keras
import tensorflow_datasets as tfds
import numpy as np
import os
import pickle
from utils import *
print(f'TF version: {tf.__version__}')


PCT = 100  # % of dataset that is loaded (0 to 100)
EPOCHS = 10
# NN = 'NN'
NN = 'RNN'  # recurrent neuronal network
MODEL_PATH = f'E:/models/tensorflow/imdb_{NN}'
TOKENIZER = f'E:/models/tensorflow/imdb_tokenizer.pickle'


try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER, 'rb') as f:
        tokenizer = pickle.load(f)

except OSError:
    # Load IMDB dataset from TensorFlow datasets
    train_ds, test_ds = tfds.load('imdb_reviews', split=[f'train[:{PCT}%]', f'test[:{PCT}%]'])
    train_ds, test_ds = tfds.as_numpy(train_ds), tfds.as_numpy(test_ds)  # convert tensor to numpy


    def extract_text_and_labels(data):
        texts = np.asarray([process_text(e['text'].decode('utf-8')) for e in data])
        labels = np.asarray([e['label'] for e in data])
        return texts, labels


    # Extract processed text and labels
    train_data, train_labels = extract_text_and_labels(train_ds)
    test_data, test_labels = extract_text_and_labels(test_ds)

    tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=20000, oov_token='<OOV>')  # out of vocabulary token
    tokenizer.fit_on_texts(train_data)  # generate tokenizer with word index

    with open(TOKENIZER, 'wb') as f:
        pickle.dump(tokenizer, f)  # save tokenizer to pickle file

    # Tokenize words
    train_data, test_data = tokenizer.texts_to_sequences(train_data), tokenizer.texts_to_sequences(test_data)
    # plot_sentence_length_distribution(train_data)  # to decide on max length


    def add_padding_to_data(data):
        return tf.keras.utils.pad_sequences(data, padding='post', maxlen=256)


    train_data = add_padding_to_data(train_data)
    test_data = add_padding_to_data(test_data)
    vocab_size = len(tokenizer.word_index)

    # Build the model
    if NN == 'NN':
        embedding_dim = int(vocab_size ** 0.25)  # embedding size should be the fourth root of vocab size

        model = keras.Sequential()
        model.add(keras.layers.Embedding(vocab_size, embedding_dim))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(embedding_dim, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
    else:
        embedding_dim = 32  # needs to be higher for RNN

        model = keras.Sequential()
        model.add(keras.layers.Embedding(vocab_size, embedding_dim))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, return_sequences=True, dropout=0.2)))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, dropout=0.2)))
        model.add(keras.layers.Dense(24, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

    model.summary()  # check model architecture

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Split into train and validation data
    split_val = len(train_data) // 4
    x_val, y_val = train_data[:split_val], train_labels[:split_val]
    x_train, y_train = train_data[split_val:], train_labels[split_val:]

    # Fit model
    model.fit(x_train, y_train, epochs=EPOCHS, batch_size=512, validation_data=(x_val, y_val), verbose=1)

    # Save the model
    model.save(MODEL_PATH)

    # Evaluate the model on test data
    model.evaluate(test_data, test_labels)


# Get word index and reverse word index
reverse_word_index = dict([(value, key) for (key, value) in tokenizer.word_index.items()])

for file in [f for f in os.listdir() if f.endswith('.txt')]:
    print('\033[1m' + file + '\033[0m')  # bold
    with open(file, encoding='utf-8') as f:
        txt = f.read()
        print('\x1B[3m' + txt + '\x1B[0m')  # italics
        processed_text = process_text(txt)
        token = [tokenizer.word_index[w] if w in tokenizer.word_index else 0 for w in processed_text.split()]
        padded_token = tf.keras.utils.pad_sequences([token], padding='post', maxlen=256)
        predict = model.predict(padded_token)
        print_prediction(predict[0][0])
        create_wordcloud(txt)
        # decoded_txt = ' '.join([reverse_word_index.get(i, '?') for i in token])  # translate token back to text
