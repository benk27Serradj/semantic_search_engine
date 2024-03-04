import pandas as pd

df = pd.read_csv('cleaned_movie_plots.csv')

# Assuming df is your DataFrame and 'Genre' is your column with the genres
# Split genres and convert to list
genres = df['Genre'].str.split(r'[,/]').tolist()

# Flatten the list
flattened_genres = [item.strip(
) for sublist in genres for item in sublist if isinstance(sublist, list)]

# Get unique genres and their counts
genre_counts = pd.Series(flattened_genres).value_counts()

# Convert the series to a DataFrame
df_genre_counts = genre_counts.reset_index()
df_genre_counts.columns = ['Genre', 'Count']

# Filter the DataFrame to only include genres that appear more than 50 times
df_genre_counts = df_genre_counts[df_genre_counts['Count'] > 50]

# Write the DataFrame to a CSV file
df_genre_counts.to_csv('genre_counts.csv', index=False)
