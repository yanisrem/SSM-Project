import numpy as np
import pandas as pd
import tensorflow as tf
import nltk
nltk.download('punkt')

def vectorize_data(data):

    def tokenize(text):
        return nltk.word_tokenize(text)
    
    dataset_tokenized = data.copy()
    dataset_tokenized["tokenized_text"]=data["Text"].apply(tokenize)

    df_tokenized_filtered = dataset_tokenized[dataset_tokenized["tokenized_text"].apply(lambda x: len(x) >= 500)]

    vocab = set()
    for token in df_tokenized_filtered["tokenized_text"]:
        vocab.update([word for word in token])

    vocab = sorted(vocab)

    voc2index = {voc: index for index, voc in enumerate(vocab)}
    index2voc = np.array(vocab)


    def voc_to_index(voc):
        voc_index = voc2index[voc] if voc in voc2index else voc2index['?']
        return voc_index

    def apply_voc_to_index_on_token(token):
        return [voc_to_index(word) for word in token]
    
    dataset_vectorized=df_tokenized_filtered.copy()
    dataset_vectorized["vectorized_text"]=df_tokenized_filtered["tokenized_text"].apply(apply_voc_to_index_on_token)

    return dataset_vectorized