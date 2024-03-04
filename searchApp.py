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
data = pd.read_csv('processed_movie_plot_data.csv')

genres = pd.read_csv('genre_counts.csv')
# -------------------------------------------------------


# Get the trained models -------------------------------
movie_tfidf_model = gensim.models.TfidfModel.load('movie_tfidf_model')
movie_lsi_model = gensim.models.LsiModel.load('movie_lsi_model')

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

    movie_index.num_best = 50

    movies_list = movie_index[query_lsi]

    # Filter movies based on genre and relevance score
    if genre is not None:
        movies_list = [movie for movie in movies_list if (
            re.search(genre, data['Genre'][movie[0]], re.IGNORECASE) or movie[1] > 0.8)]

    movies_list.sort(key=itemgetter(1), reverse=True)
    movie_names = []

    for j, movie in enumerate(movies_list):

        movie_names.append(
            {
                'Relevance': round((movie[1] * 100), 2),
                'Movie Title': data['Title'][movie[0]],
                'Movie Plot': data['Plot'][movie[0]],
                'Genre': data['Genre'][movie[0]],
                'Wikipedia Link': data['Wiki Page'][movie[0]]
            }

        )
        if j == (movie_index.num_best-1):
            break

    return pd.DataFrame(movie_names, columns=['Relevance', 'Movie Title', 'Genre', 'Wikipedia Link'])
# -------------------------------------------------------


start_time = time.time()


def main():
    st.title('Movies Search Engine')

    search_query = st.text_input("Enter your search query")
    genre = st.selectbox("Select a genre", options=genres, placeholder="All")

    results = None  # Initialize results outside the if block

    if st.button("Search"):
        results = search(search_query, genre)

    if results is not None and not results.empty:
        results['Link_Markdown'] = results.apply(
            lambda row: f"[Wikipedia Link]({row['Wikipedia Link']})", axis=1)
        html_table = results.to_html(escape=False, index=False)
        st.write(html_table, unsafe_allow_html=True)
    elif results is None:
        st.warning("Press the 'Search' button to find results.")
    else:
        st.warning("No results found.")


if __name__ == "__main__":
    main()

print(" --- %s seconds ---" % (time.time() - start_time))
