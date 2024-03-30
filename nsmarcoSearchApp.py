
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import streamlit as st
import time


@st.cache(allow_output_mutation=True)
def load_data():
    df = pd.read_csv('movies_dataset_saved.csv')
    data_plots = df['search'].tolist()
    with open("nsmarco_model.pkl", "rb") as f:
        nsmarco = pickle.load(f)
    with open("document_embeddings.pkl", "rb") as f:
        document_embeddings = pickle.load(f)
    return df, data_plots, nsmarco, document_embeddings

def semantic_search(query, data_plots, movie_data, model, document_embeddings):
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    sorted_results = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:1000]
    search_results = [(str(idx+1), movie_data.iloc[idx]['Title'], similarity) for idx, similarity in sorted_results]
    return search_results

def main():
    st.title('Movies Search Engine')
    
    df, data_plots, nsmarco, document_embeddings = load_data()
    
    search_query = st.text_input("Enter your search query")
    
    results = None  # Initialize results outside the if block
    
    if st.button("Search"):
        start_time = time.time()
        results = semantic_search(search_query, data_plots, df, nsmarco, document_embeddings)
        search_time = time.time()-start_time
        st.write(f"Search completed in {search_time:.2f} seconds.")

    if results is not None and len(results) > 0:
        st.table(results)
    elif results is None:
        st.warning("Press the 'Search' button to find results.")
    else:
        st.warning("No results found.")

if __name__ == "__main__":
    main()
