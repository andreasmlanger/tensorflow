"""
Tensorflow NNs to recognize good or bad IMDB reviews
"""

from utils import *
import keras
from keras.datasets import imdb
import tensorflow as tf
print(f'TF version: {tf.__version__}')

# NN = 'NN'  # (~87.9%)
NN = 'RNN'  # (~85.0%) recurrent neuronal network

MODEL_PATH = f'E:/models/imdb_{NN}.keras'
EPOCHS = 50 if NN == 'NN' else 5
VOCAB_SIZE = 10000

try:
    model = keras.models.load_model(MODEL_PATH)

except (ValueError, OSError):
    # Load IMDB dataset from TensorFlow datasets
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=VOCAB_SIZE)

    def add_padding_to_data(data):
        return keras.utils.pad_sequences(data, padding='post', maxlen=256)

    train_data = add_padding_to_data(train_data)
    test_data = add_padding_to_data(test_data)

    # Build the model
    if NN == 'NN':
        embedding_dim = int(VOCAB_SIZE ** 0.25)  # embedding size should be the fourth root of vocab size

        model = keras.Sequential()
        model.add(keras.layers.Embedding(VOCAB_SIZE, embedding_dim))
        model.add(keras.layers.GlobalAveragePooling1D())
        model.add(keras.layers.Dense(embedding_dim, activation='relu'))
        model.add(keras.layers.Dense(1, activation='sigmoid'))
    else:
        embedding_dim = 64  # needs to be higher for RNN

        model = keras.Sequential()
        model.add(keras.layers.Embedding(VOCAB_SIZE, embedding_dim))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, return_sequences=True, dropout=0.2)))
        model.add(keras.layers.Bidirectional(keras.layers.LSTM(embedding_dim, dropout=0.2)))
        model.add(keras.layers.Dense(64, activation='relu'))
        model.add(keras.layers.BatchNormalization())
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(1, activation='sigmoid'))

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
    model.summary()  # check model architecture
    model.evaluate(test_data, test_labels)

# Load word index and create inverted word index
word_index = keras.datasets.imdb.get_word_index()
start_char, oov_char, index_from = 1, 2, 3
inverted_word_index = dict((i + index_from, word) for (word, i) in word_index.items())
inverted_word_index[start_char] = '[START]'
inverted_word_index[oov_char] = '[OOV]'

for file in [f for f in os.listdir() if f.endswith('.txt')]:
    print('\033[1m' + file + '\033[0m')  # bold
    with open(file, encoding='utf-8') as f:
        txt = f.read()
        print('\x1B[3m' + txt + '\x1B[0m')  # italics
        processed_text = process_text(txt)
        token = [word_index[w] + index_from if w in word_index else oov_char for w in processed_text.split()]
        padded_token = keras.utils.pad_sequences([token], padding='post', maxlen=256)
        predict = model.predict(padded_token)
        print_prediction(predict[0][0])
        create_wordcloud(txt)
        # decoded_sequence = ' '.join(inverted_word_index[i] for i in token)  # translate token back to text
        # print(decoded_sequence)
