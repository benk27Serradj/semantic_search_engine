# Import -----------------------------------------------
# -------------------------------------------------------

import pandas as pd
import numpy as np
import spacy
import string
import gensim
import operator
import re

import time

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim import corpora
from gensim.similarities import MatrixSimilarity
from operator import itemgetter

from utils import text_tokenizer


import streamlit as st
# -------------------------------------------------------


# Get the processed data -------------------------------
data = pd.read_csv('cleaned_movie_plots.csv')

genres = pd.read_csv('genre_counts.csv')
# -------------------------------------------------------


# Get the trained models -------------------------------
movie_tfidf_model = gensim.models.TfidfModel.load('movie_tfidf_model_mm')
movie_lsi_model = gensim.models.LsiModel.load('movie_lsi_model_mm')

# -------------------------------------------------------

# Load the dictionary ----------------------------------
dictionary = corpora.Dictionary.load('movie_dictionary')

# -------------------------------------------------------


# Load the Similarity Index ----------------------------
movie_index = MatrixSimilarity.load('movie_index')
# -------------------------------------------------------


# Search -----------------------------------------------


def search(input_query, genre=None):

    tokenized_input = text_tokenizer(input_query)
    bow_input = dictionary.doc2bow(tokenized_input)

    query_tfidf = movie_tfidf_model[bow_input]
    query_lsi = movie_lsi_model[query_tfidf]

    movie_index.num_best = 10

    if genre is not None:
        movie_list = [
            movie for movie in movie_list if genre in data['Genre'][movie[0]]]
    else:
        movies_list = movie_index[query_lsi]

    movies_list.sort(key=itemgetter(1), reverse=True)
    movie_names = []

    for j, movie in enumerate(movies_list):

        movie_names.append(
            {
                'Relevance': round((movie[1] * 100), 2),
                'Movie Title': data['Title'][movie[0]],
                'Movie Plot': data['Plot'][movie[0]],
                'Wikipedia Link': data['Wiki Page'][movie[0]]
            }

        )
        if j == (movie_index.num_best-1):
            break

    return pd.DataFrame(movie_names, columns=['Relevance', 'Movie Title', 'Movie Plot', 'Wikipedia Link'])
# -------------------------------------------------------


start_time = time.time()


def main():
    st.title('Movies Search Engine')

    search_query = st.text_input("Enter your search query")
    genre = st.selectbox("Select a genre", options=genres)

    results = None  # Initialize results outside the if block

    if st.button("Search"):
        results = search(search_query, genre)

    if results is not None and not results.empty:
        st.table(results)
    elif results is None:
        st.warning("Press the 'Search' button to find results.")
    else:
        st.warning("No results found.")


if __name__ == "__main__":
    main()

print(" --- %s seconds ---" % (time.time() - start_time))
