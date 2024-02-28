# Import
import spacy
import string
import gensim
import operator
import re
from gensim import corpora
from gensim.similarities import MatrixSimilarity
from operator import itemgetter
import streamlit as st # for the UI

import pandas as pd
import numpy as np
import matplotlib as plt
import seaborn as sns


import nltk
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


data = pd.read_csv('archive/wiki_movie_plots_deduped.csv')


# Create a list of stopwords
stop_words = set(stopwords.words('english'))

lemmatizer = WordNetLemmatizer()






def nltk_tokenizer(text):
    
    # Remove any characters that are not uppercase letters, lowercase letters, or white space character.
    cleaned_text = re.sub(r'[^A-Za-z\s]', '', text) 
    
    # Replace conecutive spaces with a single space.
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    
    # Creating token objects
    tokens = word_tokenize(cleaned_text)
    
    
    lowercase_tokens = [token.lower() for token in tokens]
    
    filtered_tokens = [token for token in lowercase_tokens if token not in stop_words]
    
    lemmatized_tokens = [lemmatizer.lemmatize(token) for token in filtered_tokens]
    
    return lemmatized_tokens
    


data['plot_tokenized'] = data['Plot'].map(lambda x : nltk_tokenizer(x))

movie_tokenized = data['plot_tokenized']


dictionary = corpora.Dictionary(movie_tokenized)


# Creating a list of lists
dict_tokens = [
    [
        [dictionary[key]
         , dictionary.token2id[dictionary[key]]]
        for key, value in dictionary.items()
        if key <= 50
    ]
]


corpus = [dictionary.doc2bow(desc) for desc in movie_tokenized]

word_frequencies = [[(dictionary[id], frequency) for id, frequency in line] for line in corpus[0:3]]

movie_tfidf_model = gensim.models.TfidfModel(corpus, id2word=dictionary)

movie_lsi_model = gensim.models.LsiModel(movie_tfidf_model[corpus], id2word=dictionary, num_topics=400)

# Serialize the output of the model

gensim.corpora.MmCorpus.serialize('movie_tfidf_model_mm', movie_tfidf_model[corpus])

gensim.corpora.MmCorpus.serialize('movie_lsi_model_mm',movie_lsi_model[movie_tfidf_model[corpus]])


# Load the previously serialized models back to memory.
# This allows you to use the preprocessed without having to remcompute the transformers


movie_tfidf_corpus = gensim.corpora.MmCorpus('movie_tfidf_model_mm')
movie_lsi_corpus = gensim.corpora.MmCorpus('movie_lsi_model_mm')

movie_index = MatrixSimilarity(movie_lsi_corpus, num_features=movie_lsi_corpus.num_terms)




def search(input_query):
    
    tokenized_input = nltk_tokenizer(input_query)
    bow_input = dictionary.doc2bow(tokenized_input)
    
    query_tfidf = movie_tfidf_model[bow_input]
    query_lsi = movie_lsi_model[query_tfidf]
    
    movie_index.num_best = 10
    
    movies_list = movie_index[query_lsi]
    
    
    movies_list.sort(key=itemgetter(1), reverse=True)
    movie_names = []
    
    for j, movie in enumerate(movies_list):

        movie_names.append (
            {
                'Relevance': round((movie[1] * 100),2),
                'Movie Title': data['Title'][movie[0]],
                'Movie Plot': data['Plot'][movie[0]],
                'Wikipedia Link' : data['Wiki Page'][movie[0]]
            }

        )
        if j == (movie_index.num_best-1):
            break

    return pd.DataFrame(movie_names, columns=['Relevance','Movie Title','Movie Plot', 'Wikipedia Link'])



def main():
    st.title('Movies Search Engine')
    
    search_query = st.text_input("Enter your search query")
    
    if st.button("Search"):
        results = search(search_query)
    
    if not results.empty:
        st.table(results)
    else:
        st.warning("No results found.")

if __name__ == "__main__":
    main()