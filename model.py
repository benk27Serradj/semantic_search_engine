
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import pickle

df = pd.read_csv('movies_dataset_saved.csv')
data_plots = df['search'].tolist()


# Load the model
with open("nsmarco_model.pkl", "rb") as f:
    nsmarco = pickle.load(f)

# Load the document embeddings
with open("document_embeddings.pkl", "rb") as f:
    document_embeddings = pickle.load(f)

  
def semantic_search(query, data=data_plots, movie_data=df):
    query_embedding = nsmarco.encode([query])[0]

    similarities = cosine_similarity([query_embedding], document_embeddings)[0]  # type: ignore

    sorted_results = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)

    search_results = [(str(idx+1), movie_data.iloc[idx]['Title'], similarity) for idx, similarity in sorted_results]
    return search_results


query = 'decaprio in a big boat with an older lady going from uk to usa'

r1 = semantic_search(query)
print(r1[:10])
