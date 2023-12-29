import pandas as pd
import numpy as np
from gensim import corpora, models
import gensim

class LDA:
    def __init__(self, num_topics, random_state):
        self.num_topics=num_topics
        self.random_state=random_state
    
    def run(self, X):
        X_str = X.astype(str)
        dictionary = corpora.Dictionary(X_str)
        corpus = [dictionary.doc2bow(text) for text in X_str]
        ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=self.num_topics, id2word = dictionary, random_state=self.random_state)
        topics_terms = ldamodel.state.get_lambda() 
        topics_terms_proba = np.apply_along_axis(lambda x: x/x.sum(),1,topics_terms)
        words = [ldamodel.id2word[i] for i in range(topics_terms_proba.shape[1])]
        topics_matrix=pd.DataFrame(topics_terms_proba,columns=words)
        topic_by_word=topics_matrix.idxmax()
        topic_by_word=pd.DataFrame({'word':topic_by_word.index, 'topic_index':topic_by_word.values})
        word_to_topic = topic_by_word.set_index('word')['topic_index'].to_dict()
        def map_topic_to_words(row):
            return np.vectorize(word_to_topic.get)(row)
        Z = np.apply_along_axis(map_topic_to_words, axis=1, arr=X_str)
        return Z




