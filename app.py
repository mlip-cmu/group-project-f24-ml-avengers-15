from flask import Flask, jsonify
import kafka_server_apis as kafka
import utils as utils
import pickle
import os
import traceback
from online_evaluation import calculate_precision_at_k, get_user_relevant_movies, consume_kafka_data

app = Flask(__name__)

SERVER_IP = os.getenv("TEAM_15_SERVER_IP")
MOVIES_CSV = os.path.join("data", "movies.csv")

kafka_server = kafka.KafkaServerApi(server_ip=SERVER_IP)

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
    
@app.route('/evaluate_snapshot/<int:k>', methods=['GET'])
def evaluate_snapshot(k: int):
    try:
        # Consume data from Kafka for the past 24 hours
        snapshot_df = consume_kafka_data(duration=86400) 
        user_relevant_movies = get_user_relevant_movies(snapshot_df)
        user_recommendations = {user_id: recommend_movies(user_id) for user_id in user_relevant_movies.keys()}
        precision = calculate_precision_at_k(user_recommendations, user_relevant_movies, k)
        return jsonify({"Precision@K": precision})
    
    except Exception as e:
        print(f"An error occurred while calculating Precision@K: {e}")
        traceback.print_exc()
        return jsonify({"Precision@K": None})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8082)
