from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
import os
import string
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

TABLE = str.maketrans('', '', string.punctuation)  # !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~


def process_text(x):
    x = x.lower().replace(',', ', ').replace('.', '. ').replace('-', '- ').replace('/', '/ ')
    x = BeautifulSoup(x, features='lxml').get_text()
    return ' '.join([tw for w in x.split() if (tw := w.translate(TABLE)) not in STOPWORDS])


def plot_sentence_length_distribution(arr):
    plt.ylabel('Number of words')
    plt.plot(sorted([len(s) for s in arr]))
    plt.show()


def create_wordcloud(text):
    cloud = WordCloud(stopwords=STOPWORDS, width=1500, height=500, background_color='#0e253a').generate(text)
    plt.figure(figsize=(9, 3))
    plt.imshow(cloud, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout(pad=0)
    plt.show()


def print_prediction(x):
    if x > 0.5:
        print('This review is\033[36m positive\033[38m ({:2.0f}%)'.format(100 * x))
    else:
        print('This review is\033[31m negative\033[38m ({:2.0f}%)'.format(100 * (1 - x)))
