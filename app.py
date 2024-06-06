# streamlit run app.py --client.showErrorDetails=false
from calendar import c
from turtle import color
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle
import streamlit as st
import time
from langdetect import detect, DetectorFactory, LangDetectException

# Ensure consistent result for language detection
DetectorFactory.seed = 0

def is_mixed_language(text):
    words = text.split()
    detected_languages = set()
    for word in words:
        try:
            detected_language = detect(word)
            detected_languages.add(detected_language)
        except LangDetectException:
            continue
    is_there_arabic = "ar" in detected_languages
    result = is_there_arabic and len(detected_languages) > 1
    return result

@st.cache_data
def load_data(language):
    df = pd.read_csv('movies_dataset_saved.csv')
    if language == "ar":
        data_plots = df['Plot_ar'].tolist()
    elif language == "mixed":
        data_plots = df['search_Mixt'].tolist()
    else:
        data_plots = df['search'].tolist()
    return df, data_plots

@st.cache_resource
def load_models(language):
    if language == "ar":
        with open("Mixt_MiniLm.pkl", "rb") as f:
            model = pickle.load(f)
        with open("arabic_embeddings.pkl", "rb") as f:
            document_embeddings = pickle.load(f)
    elif language == "mixed":
        with open("Mixt_MiniLm.pkl", "rb") as f:
            model = pickle.load(f)
        with open("Mixt_embeddings.pkl", "rb") as f:
            document_embeddings = pickle.load(f)
    else:
        with open("English_MiniLm.pkl", "rb") as f:
            model = pickle.load(f)
        with open("English_embeddings.pkl", "rb") as f:
            document_embeddings = pickle.load(f)
    return model, document_embeddings

def semantic_search(query, dataset, data_plots, model, document_embeddings):
    query_embedding = model.encode([query])[0]
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    sorted_results = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)[:10]
    search_results = [(dataset.iloc[idx]['Title'], similarity) for idx, similarity in sorted_results]
    return search_results

st.set_page_config(page_title='Movie Search Engine', page_icon='🎬', layout='centered')

st.markdown("""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Lobster&family=Raleway:wght@400;700&family=Roboto:wght@400;700&display=swap');

        /* Hide Streamlit's main menu, header, and footer */
        header {visibility: hidden;}
        footer {visibility: hidden;}

        .stApp {
            background: #00111C; /* Dark Blue */
            background: -webkit-linear-gradient(to bottom, #00111C, #00406C); /* W3C, IE 10+/ Edge, Chrome 26+, Firefox 26+, Opera 12+, Safari 7+ */
            background: linear-gradient(to bottom, #00111C, #00406C); /* Standard syntax */
        }
        /* Additional styling */
        .title {
            font-size: 3em;
            font-weight: bold;
            color: #FFD700; /* Gold */
            text-align: center;
            margin-top: 20px;
            animation: fadeIn 2s;
            font-family: 'Lobster', cursive; /* Change font family */
        }
        .description {
            font-size: 1.2em;
            color: #FFD700; /* Gold */
            text-align: center;
            margin-bottom: 20px;
            animation: fadeIn 3s;
            font-family: 'Raleway', sans-serif; /* Change font family */
        }
        .input-container {
            text-align: center;
            margin-top: 20px;
            animation: fadeIn 4s;
        }
        .search-button {
            background-color: #FFD700; /* Gold */
            color: #00111C; /* Dark Blue */
            font-size: 1.2em;
            padding: 10px 20px;
            border: none;
            border-radius: 5px;
            cursor: pointer;
            transition: background-color 0.3s ease;
            animation: fadeIn 5s;
            font-family: 'Roboto', sans-serif; /* Change font family */
        }
        .search-button.search-button:hover {
            background-color: #FFD700; /* Gold */
            color: #00111C; /* Dark Blue */
        }
        .results {
            margin-top: 20px;
            animation: fadeIn 6s;
            color: #FFD700; /* Gold */
        }
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        .stTextInput>div>div>input {
            background-color: #002E4E; /* Darker Blue */
            border: 2px solid #FFD700; /* Gold */
            border-radius: 5px;
            padding: 10px;
            font-size: 1.1em;
            width: 1000px; 
            color: #FFFFFF; /* White */
            font-family: 'Roboto', sans-serif; /* Change font family */
        }
        .stTextInput>div>div>input::placeholder {
            color: #FFD700; /* Gold */
        }
        .stTable tbody tr td {
            color: #FFD700; /* Gold text for table data */
        }
        .custom-table th {
            color: #FFD700; /* Gold text for table headers */
            background-color: #003A61; /* Darker Blue background for headers */
            font-family: 'Roboto', sans-serif; /* Change font family */
        }
        .custom-table td {
            color: #FFD700; /* Gold text for table data */
            background-color: #00111C; /* Dark Blue background for table rows */
            font-family: 'Roboto', sans-serif; /* Change font family */
        }
        </style>
    """, unsafe_allow_html=True)
    
def main():
    st.markdown('<div class="title">🎬 Movie Search Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="description">Welcome to the Movie Search Engine. Type in the movie name or related keywords to find information about your favorite movies.</div>', unsafe_allow_html=True)

    # Input field for search query
    st.markdown('<div class="input-container">', unsafe_allow_html=True)
    search_query = st.text_input("", placeholder="Find your movie")
    st.markdown('</div>', unsafe_allow_html=True)

    # Placeholder for search results
    results = None

    # Search button with a bit of spacing
    if st.button("Search", key="search_button", help="Click to search"):
        if not search_query.strip():
            st.warning("Please enter a search query.")
        else:
            start_time = time.time()
            
            # Detect the language of the search query
            if is_mixed_language(search_query):
                language = "mixed"
            else:
                language = detect(search_query)
            
            # Load necessary data and model
            df, data_plots = load_data(language)
            model, document_embeddings = load_models(language)
            
            # Perform semantic search
            results = semantic_search(search_query, dataset=df, data_plots=data_plots, model=model, document_embeddings=document_embeddings)
            
            search_time = time.time() - start_time
            st.success(f"Search completed in {search_time:.2f} seconds.")
    
    # Display results in a more user-friendly manner
    if results is not None and len(results) > 0:
        st.markdown('<div class="results">', unsafe_allow_html=True)
        st.subheader("Search Results:")
        # Display only the top 10 results
        results_df = pd.DataFrame(results, columns=["Title", "Score"])
        st.markdown("<style>.custom-table tbody tr td {color: white;}</style>", unsafe_allow_html=True)  # Ensure results text is white
        st.table(results_df.style.set_table_styles(
            [{'selector': 'th', 'props': [('color', 'white'), ('background-color', '#003A61')]},
             {'selector': 'td', 'props': [('color', 'white'), ('background-color', '#00111C')]}]
        ))
        st.markdown('</div>', unsafe_allow_html=True)
    elif results is None:
        st.info("Press the 'Search' button to find results.")
    else:
        st.info("No results found.")

        
if __name__ == "__main__":
    main()
