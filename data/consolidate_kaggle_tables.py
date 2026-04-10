import os
import pandas as pd 

## Read in raw kaggle data -------------------------------------------
movies = pd.read_csv('c:/Users/claro/OneDrive - Cornell University/Quantitative Modeling/Rotten Tomatoes/data/rotten_tomatoes_movies.csv')
reviews = pd.read_csv('c:/Users/claro/OneDrive - Cornell University/Quantitative Modeling/Rotten Tomatoes/data/rotten_tomatoes_movie_reviews.csv') 


## Creating data sets for project  -----------------------------------
movies_consol = movies[['id', 'title', 'audienceScore', 'tomatoMeter', 'rating', 'genre','runtimeMinutes']]
reviews_consol = reviews[['id', 'reviewId', 'publicatioName', 'criticName', 'isTopCritic', 'originalScore','reviewState','creationDate']]

master_table = pd.merge(movies_consol, reviews_consol, on='id', how='outer')

reviews = reviews[['reviewId', 'criticName','reviewText']]

movies = movies[['id', 'title', 'audienceScore', 'tomatoMeter', 'rating', 'genre','runtimeMinutes', 'director', 'writer']]


## exporting created data sets to csv --------------------------------
output_dir = os.path.dirname(__file__)
movies.to_csv(os.path.join(output_dir, 'movies.csv'), index=False)
reviews.to_csv(os.path.join(output_dir, 'reviews.csv'), index=False)
master_table.to_csv(os.path.join(output_dir, 'master_table.csv'), index=False)