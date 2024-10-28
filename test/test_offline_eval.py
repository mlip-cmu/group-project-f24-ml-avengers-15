import unittest
import pandas as pd
import numpy as np
from io import StringIO

class TestMovieRatingAnalysis(unittest.TestCase):

    def setUp(self):
        # Sample data for testing
        self.sample_data = StringIO("""user_id,age,occupation,gender,movie_id,rating
            1,25,student,M,movie1,4
            2,30,engineer,F,movie2,3
            1,25,student,M,movie3,5
            3,40,teacher,F,movie1,2
        """)
        self.df = pd.read_csv(self.sample_data)

    def test_data_loading(self):
        self.assertIsInstance(self.df, pd.DataFrame)
        self.assertEqual(len(self.df), 4)
        self.assertEqual(list(self.df.columns), ['user_id', 'age', 'occupation', 'gender', 'movie_id', 'rating'])

    def test_data_types(self):
        self.assertEqual(self.df['user_id'].dtype, np.int64)
        self.assertEqual(self.df['age'].dtype, np.int64)
        self.assertEqual(self.df['occupation'].dtype, object)
        self.assertEqual(self.df['gender'].dtype, object)
        self.assertEqual(self.df['movie_id'].dtype, object)
        self.assertEqual(self.df['rating'].dtype, np.int64)

    def test_unique_users(self):
        unique_users = self.df['user_id'].nunique()
        self.assertEqual(unique_users, 3)

    def test_unique_movies(self):
        unique_movies = self.df['movie_id'].nunique()
        self.assertEqual(unique_movies, 3)

    def test_rating_range(self):
        min_rating = self.df['rating'].min()
        max_rating = self.df['rating'].max()
        self.assertGreaterEqual(min_rating, 1)
        self.assertLessEqual(max_rating, 5)

    def test_gender_values(self):
        valid_genders = ['M', 'F']
        self.assertTrue(self.df['gender'].isin(valid_genders).all())

    def test_age_range(self):
        min_age = self.df['age'].min()
        max_age = self.df['age'].max()
        self.assertGreaterEqual(min_age, 0)
        self.assertLess(max_age, 100)

    def test_group_by_user(self):
        user_ratings = self.df.groupby('user_id')['rating'].mean()
        self.assertEqual(len(user_ratings), 3)
        self.assertAlmostEqual(user_ratings[1], 4.5, places=1)

    def test_group_by_movie(self):
        movie_ratings = self.df.groupby('movie_id')['rating'].mean()
        self.assertEqual(len(movie_ratings), 3)
        self.assertAlmostEqual(movie_ratings['movie1'], 3.0, places=1)

    def test_occupation_distribution(self):
        occupation_counts = self.df['occupation'].value_counts()
        self.assertEqual(len(occupation_counts), 3)
        self.assertEqual(occupation_counts['student'], 2)

if __name__ == '__main__':
    unittest.main()