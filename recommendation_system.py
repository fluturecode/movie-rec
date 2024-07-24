import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Load the ratings dataset
ratings = pd.read_csv('data/ml-100k/u.data', sep='\t', names=['userId', 'movieId', 'rating', 'timestamp'])

# Load the movies dataset
movies = pd.read_csv('data/ml-100k/u.item', sep='|', names=['movieId', 'title'], usecols=[0, 1], encoding='latin-1')

# Merge the datasets on 'movieId'
data = pd.merge(ratings, movies, on='movieId')

# Create a user-item matrix
user_item_matrix = data.pivot_table(index='userId', columns='title', values='rating')
user_item_matrix.fillna(0, inplace=True)

# Compute the cosine similarity matrix
item_similarity = cosine_similarity(user_item_matrix.T)

# Function to get movie recommendations
def get_movie_recommendations(movie_name, similarity_matrix, movie_titles):
    # Find the index of the movie
    idx = movie_titles[movie_titles == movie_name].index[0]
    
    # Get the similarity scores for all movies with that movie
    sim_scores = list(enumerate(similarity_matrix[idx]))
    
    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # Get the scores of the 10 most similar movies
    sim_scores = sim_scores[1:11]
    
    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]
    
    # Return the top 10 most similar movies
    return movie_titles.iloc[movie_indices]

# Get movie titles
movie_titles = data['title'].drop_duplicates()

# Example usage
recommendations = get_movie_recommendations('Toy Story (1995)', item_similarity, movie_titles)
print("Recommended movies for 'Toy Story (1995)':")
print(recommendations)