import pandas as pd
from kafka import KafkaConsumer
import json
import pandas as pd

TOPIC_NAME = "movielog15"

def consume_kafka_data(topic: str, duration: int) -> pd.DataFrame:
    consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='rating-group',
    value_deserializer=lambda x: x.decode('utf-8')
)

    end_time = time.time() + duration
    data = []

    for message in consumer:
        if time.time() > end_time:
            break
        data.append(message.value)

    consumer.close()
    return pd.DataFrame(data)

# Calculate Precision@K
def calculate_precision_at_k(user_recommendations, user_relevant_movies, k):
    total_precision = 0
    total_users = 0

    for user_id, rec_movies in user_recommendations.items():
        relevant_movies = user_relevant_movies.get(user_id, set())
        if relevant_movies:
            top_k_recommendations = set(rec_movies[:k])  # top K recommendations
            relevant_count = len(top_k_recommendations.intersection(relevant_movies))
            precision = relevant_count / k  
            total_precision += precision
            total_users += 1

    return total_precision / total_users if total_users > 0 else 0  # Average Precision@K over users

def get_user_relevant_movies(df, rating_threshold=4):
    return df[df['rating'] >= rating_threshold].groupby('user_id')['movie_id'].apply(set).to_dict()
