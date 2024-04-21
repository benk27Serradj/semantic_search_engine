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

    movie_index.num_best = 20

    movies_list = movie_index[query_lsi]

    movies_list.sort(key=itemgetter(1), reverse=True)
    movie_names = []

    for j, movie in enumerate(movies_list):
        wiki_url = data['Wiki Page'][movie[0]]
        movie_title = data['Title'][movie[0]]
        poster_url = None
        # poster_url = get_poster_url(wiki_url, movie_title)


        movie_names.append(
            {
                'Movie Title': movie_title,
                'Relevance': round((movie[1] * 100), 2),
                'Genre': data['Genre'][movie[0]],
                'Wikipedia Link': wiki_url,
                'Poster Link': poster_url
            }

        )
        if j == (movie_index.num_best-1):
            break

    return pd.DataFrame(movie_names, columns=['Relevance', 'Movie Title', 'Genre', 'Wikipedia Link', 'Poster Link'])
# -------------------------------------------------------

from bs4 import BeautifulSoup
import requests

def get_poster_url(wiki_url, movie_title):
    response = requests.get(wiki_url)
    soup = BeautifulSoup(response.text, 'html.parser')

    soup = soup.find_all('a', href=True)
    for link in soup:
        if any(word in link['href'] for word in movie_title.split()):
            return link['href']
            break

    
    return None

start_time = time.time()


def main():
    st.title('Movies Search Engine')

    search_query = st.text_input("Enter your search query")

    results = None  # Initialize results outside the if block

    if st.button("Search") or search_query:
        with st.spinner(text='In progress'):
            results = search(search_query)
            st.success('Done') 


    if results is not None and not results.empty:
        st.dataframe(results , column_config={
            'Relevance': {'format': '{:.2f}%'},
            'Wikipedia Link': st.column_config.LinkColumn("wiki page"),
            'Poster Link' : st.column_config.ImageColumn("Poster")
        })
    elif results is None:
        st.warning("Press the 'Search' button to find results.")
    else:
        st.warning("No results found.")


if __name__ == "__main__":
    main()

print(" --- %s seconds ---" % (time.time() - start_time))
