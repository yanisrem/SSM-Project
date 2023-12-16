import pandas as pd
import numpy as np
import nltk
nltk.download('punkt')
from sklearn.feature_extraction.text import CountVectorizer

def pre_processing_data(data, min_number_words):

    def tokenize(text):
        return nltk.word_tokenize(text)
    
    dataset_tokenized = data.copy()
    dataset_tokenized["tokenized_text"]=data["Text"].apply(tokenize)
    df_tokenized_filtered = dataset_tokenized[dataset_tokenized["tokenized_text"].apply(lambda x: len(x) >= min_number_words)]

    count_vectorizer = CountVectorizer()
    text_matrix = count_vectorizer.fit_transform(df_tokenized_filtered['Text'])
    df_X = pd.DataFrame(text_matrix.toarray(), columns=count_vectorizer.get_feature_names_out())

    return df_tokenized_filtered, df_X
