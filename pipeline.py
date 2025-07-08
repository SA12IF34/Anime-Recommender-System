import numpy as np
import pandas as pd
import joblib


class AnimeRecommender:

  anime_df = None

  def __init__(self, anime_views_df_path, scaler_path, model_path, vectorizer_path, anime_df_path=None):
    assert anime_views_df_path is not None
    assert scaler_path is not None
    assert model_path is not None
    assert vectorizer_path is not None

    self.anime_views_df = pd.read_csv(anime_views_df_path)

    self.model = joblib.load(model_path)
    self.vectorizer = joblib.load(vectorizer_path)
    self.feature_names = self.vectorizer.get_feature_names_out()
    self.scaler = joblib.load(scaler_path)

    if anime_df_path:
      self.anime_df = pd.read_csv(anime_df_path)


  def set_df(self, anime_views_df):
    self.anime_views_df = anime_views_df


  def process_profile(self, profile):
    scaled_profile = self.scaler.transform(profile)

    return scaled_profile


  def make_profile(self, seen_anime_ids, ratings, anime_df_path=None):
    if self.anime_df is None:
      raise AssertionError('Anime dataframe path was provided on initialization, please provide when calling make_profile')

    profile = np.zeros(len(self.feature_names))

    for anime_id in seen_anime_ids:
      anime = self.anime_df[self.anime_df['anime_id'] == anime_id]
      if len(anime) == 1:
        genre_values = anime[self.feature_names][0]

        profile += genre_values

    profile = self.scaler.transform(profile)


    return profile

  def recommend(self, profile, seen_anime_ids, rcmd_size=20):
    label = self.model.predict(profile)[0]

    condition_1 = (self.anime_views_df['labels'] == label)
    condition_2 = (~self.anime_views_df['anime_id'].isin(seen_anime_ids))

    recommendation_dataset = self.anime_views_df[condition_1 & condition_2].sort_values('views', ascending=False)

    return recommendation_dataset.head(rcmd_size)['anime_id'].values

