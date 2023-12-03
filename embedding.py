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
    else:
        wv = api.load(embedding_model, return_path=True)

# ------------------- Methods ------------------- #

def get_embedding(word):
    """
    Returns the embedding of a word
    :param word: word to get the embedding of
    :return: embedding of the word
    """
    global wv 
    if word in wv:
        return wv[word]
    else:
        print(f"Word '{word}' not found in the vocabulary.")
        return None  

def get_sentence_embedding(sentence):
    """
    Returns the embedding of a sentence
    :param sentence: sentence to get the embedding of
    :return: embedding of the sentence
    """
    words = sentence.split()
    print(words)
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
    load_embedding_model()                          # load the model (1.5 GB))
    print(get_sentence_embedding("I am a sentence")) 

# ------------------- End of File ------------------- #