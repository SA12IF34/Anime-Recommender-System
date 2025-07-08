import numpy as np
import pandas as pd
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
import string
from sklearn.feature_extraction.text import CountVectorizer
import joblib

import requests



base_path = ''

anime_df = pd.read_csv('anime.csv')
rating_df = pd.read_csv('rating.csv')

###
# Exploratory Data Analysis
###

# Anime Dataset
print("Sample Data: \n", anime_df.head())
print()
print(anime_df.info())

print("Anime Dataframe Shape: ", anime_df.shape)
print("Number of unique samples: ", len(anime_df['anime_id'].unique()))

#Rating Dataset
print("Sample Data: \n", rating_df.head())
print()
print(rating_df.info())

print("Rating Dataframe Shape: ", rating_df.shape)

###
# Data Preprocessing
###

# Fill anime data with MAL API
# https://myanimelist.net/apiconfig/references/api/v2
# https://myanimelist.net/apiconfig/references/authorization

access_token = ""

def update_anime():

  for idx, row in anime_df.iterrows():
    if True in row.isnull().values:
      response = requests.get(f'https://api.myanimelist.net/v2/anime/{row["anime_id"]}?fields=genres,mean,num_list_users,media_type', headers={
        'Authorization': 'Bearer '+access_token
        })
      print(response)
      if response.status_code == 404:
        continue
      if response.status_code == 401:
        print('Break Break')
        break
      response = response.json()

      anime_df.loc[anime_df['anime_id'] == row['anime_id'], 'rating'] = response['mean']
      anime_df.loc[anime_df['anime_id'] == row['anime_id'], 'members'] = response['num_list_users']
      anime_df.loc[anime_df['anime_id'] == row['anime_id'], 'type'] = response['media_type'].capitalize() if response['media_type'] == 'movie' else response['media_type'].upper()
      anime_df.loc[anime_df['anime_id'] == row['anime_id'], '']
      anime_df.loc[anime_df['anime_id'] == row['anime_id'], '']
      genres = []
      for g in response['genres']:
        genres.append(g['name'])

      anime_df.loc[anime_df['anime_id'] == row['anime_id'], 'genre'] = ", ".join(genres)

update_anime()

anime_df.dropna(inplace=True)

# Processing Genres 

def process_genres(genre):
  genre_list = genre.split(', ')
  new_list = []
  for genre in genre_list:
    if genre not in [',']:
      new_list.append('-'.join(genre.split(' ')).lower())

  return ' '.join(new_list)

anime_df['genre'] = anime_df['genre'].apply(process_genres)

vectorizer = CountVectorizer(ngram_range=(1, 1), token_pattern=r'\b[a-zA-Z-]+\b')
vectorized_genres = vectorizer.fit_transform(anime_df['genre'])

print("Genre labels: \n", vectorizer.get_feature_names_out())
print()
print("Sample genres row: \n", vectorized_genres.toarray()[0])

genre_df = pd.DataFrame(vectorized_genres.toarray(), columns=vectorizer.get_feature_names_out())
print(genre_df.head())

anime_df = pd.merge(anime_df, genre_df, left_index=True, right_index=True)

###
# Construct User Profiles
###

genre_labels = vectorizer.get_feature_names_out()

def make_profiles():

  users = rating_df['user_id'].unique()

  profiles = []
  print(f"Total users number: {users.shape[0]}")
  i=1
  for user in users:
    rated_animes = rating_df[rating_df['user_id'] == user]

    rating_sum = np.zeros(len(genre_labels))
    genre_sum = np.zeros(len(genre_labels))

    for idx, rated_anime in rated_animes.iterrows():
      anime = anime_df[anime_df['anime_id'] == rated_anime['anime_id']]
      genres = anime[genre_labels].values
      if len(genres) == 0:
        continue
      genres = genres[0]

      genre_sum += genres

    profiles.append([user, *genre_sum.tolist()])

    print(f"Done user {i}")

    i+=1


  print(f"Done all {i} users!")

  profiles_df = pd.DataFrame(profiles, columns=['user_id', *list(genre_labels)])

  return profiles_df

user_profiles = make_profiles()


# Save Data

user_profiles.to_csv(base_path+'profiles.csv', index=False)
anime_df.to_csv(base_path+'processed_anime_df.csv', index=False)

joblib.dump(vectorizer, base_path+'vectorizer.joblib')