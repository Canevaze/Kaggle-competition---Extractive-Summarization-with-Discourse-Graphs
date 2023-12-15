# ------------------- Imports ------------------- #

import numpy as np
import gensim.downloader as api

# ------------------- Globals ------------------- #

embedding_model = 'word2vec-google-news-300'

# -------------------- Model -------------------- #

# Loading the model

def load_embedding_model():
    """
    Loads the model if it is not in cache yet
    :return: None
    """
    global wv
    if embedding_model not in api.info()['models']:
        wv = api.load(embedding_model)
        print("Model loaded from cache.")
    else:
        wv = api.load(embedding_model)
        print("Model loaded from internet.")


# ------------------- Methods ------------------- #

def get_embedding(word,pprint):
    """
    Returns the embedding of a word
    :param word: word to get the embedding of
    :return: embedding of the word
    """
    if word in wv:
        return wv[word]
    else:
        if pprint:
            print(f"Word '{word}' not found in the vocabulary.")
        return None

def is_in_vocabulary(word):
    """
    Returns whether a word is in the vocabulary
    :param word: word to check
    :return: True if the word is in the vocabulary, False otherwise
    """
    return word in wv

def get_sentence_embedding(sentence,pprint):
    """
    Returns the embedding of a sentence
    :param sentence: sentence to get the embedding of
    :return: embedding of the sentence
    """
    words = sentence.split()
    sentence_embedding = np.zeros(300)
    m = 0
    for word in words:     
        word = word.strip('.,?!"\'').lower()        # remove punctuation and lower case
        if word =='' :
            pass
        else :
            word_embedding = get_embedding(word,pprint)
            if word_embedding is not None:
                sentence_embedding += word_embedding
                m+=1
    if m == 0:
        return np.zeros(300)
    else:
        return sentence_embedding/m

# ------------------- Testing ------------------- #

if __name__ == "__main__":
    load_embedding_model()                          # load the model (1.5 GB))
    print(get_sentence_embedding("I am a sentence",True)) 

# ------------------- End of File ------------------- #