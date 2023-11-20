# ------------------- Imports ------------------- #

import numpy as np
import gensim.downloader as api

# ------------------- Globals ------------------- #

embedding_model = 'word2vec-google-news-300'

# -------------------- Model -------------------- #

# Loading the model
def load_embedding_model():
    """
    Loads the model
    :return: None
    """
    global wv
    wv = api.load(embedding_model)

# ------------------- Methods ------------------- #

def get_embedding(word):
    """
    Returns the embedding of a word
    :param word: word to get the embedding of
    :return: embedding of the word
    """
    return wv[word]

def get_sentence_embedding(sentence):
    """
    Returns the embedding of a sentence
    :param sentence: sentence to get the embedding of
    :return: embedding of the sentence
    """
    words = sentence.split()
    sentence_embedding = np.zeros(300)
    for word in words:     
        word = word.strip('.,?!"\'').lower()        # remove punctuation
        try:
            word_embedding = get_embedding(word)
            sentence_embedding += word_embedding
        except KeyError:
            continue
    return sentence_embedding/len(words)

# ------------------- Testing ------------------- #

if __name__ == "__main__":
    load_embedding_model()                          # load the model (1.5 GB)
    print(get_sentence_embedding("I am a sentence")) 

# ------------------- End of File ------------------- #