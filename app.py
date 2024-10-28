from flask import Flask, jsonify
import kafka_server_apis as kafka
import utils as utils
import pickle
import os
import traceback

app = Flask(__name__)

SERVER_IP = os.getenv("TEAM_15_SERVER_IP")
MOVIES_CSV = os.path.join("data", "movies.csv")

#kafka_server = kafka.KafkaServerApi(server_ip=SERVER_IP)

with open(os.path.join("models", "SVD_movie_recommender.pkl"), 'rb') as f:
    model = pickle.load(f)

ratings_file = 'data/extracted_ratings.csv'
train_df, val_df = utils.prepare_data_csv(ratings_file)
train_data, valid_data = utils.prepare_data_model(train_df, val_df)
all_movies_list = train_df['movie_id'].unique().tolist()
user_movie_list = train_df.groupby('user_id')['movie_id'].apply(set).to_dict()


def recommend_movies(user_id):
    try:
        recommendations = utils.predict(model, user_id, all_movies_list, user_movie_list)

        return ','.join(recommendations)
    except Exception as e:
        print(f"An error occurred while fetching user {user_id} info: {e}")

        traceback.print_exc()

        return []

@app.route('/recommend/<int:user_id>', methods=['GET'])
def recommend(user_id):
    try:
        recommendations = recommend_movies(user_id)

        return jsonify(recommendations)
    except Exception as e:
        print(f"An error occurred while generating recommendations: {e}")
        
        traceback.print_exc()
        
        return []


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
