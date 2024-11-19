from kafka import KafkaConsumer
import re
import csv
import pandas as pd
from datetime import datetime
import os


def consume_kafka_logs(limit=100000):
    TOPIC_NAME = "movielog15"
    rate_pattern = re.compile(r'^.*?,\d+,GET /rate/.*?=\d+$')

    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='latest',
        enable_auto_commit=True,
        group_id='rating-group',
        value_deserializer=lambda x: x.decode('utf-8')
    )

    message_count = 0
    all_data = []

    for message in consumer:
        line = message.value

        if rate_pattern.match(line):
            message_timestamp = message.timestamp
            message_time = datetime.fromtimestamp(message_timestamp / 1000.0)
            time_str = message_time.strftime('%Y-%m-%dT%H:%M:%S')
            
            parts = line.strip().split(',')
            
            if len(parts) < 3:
                continue

            try:
                userid = parts[1]
                request = parts[2].strip()
                
                if not request.startswith('GET /rate/'):
                    continue
                
                movie_rating = request[len('GET /rate/'):].split('=')
                
                if len(movie_rating) < 2:
                    continue
                
                movieid = movie_rating[0]
                movie_info = movie_rating[0].split('+')
                rating = movie_rating[1]
                
                year = movie_info[-1]  
                movietitle = ' '.join(movie_info[:-1])

                all_data.append({
                    'user_time': time_str,
                    'user_id': userid,
                    'movie_id': movieid,
                    'movie_title': movietitle,
                    'year': year,
                    'rating': rating
                })
                
                message_count += 1

                if message_count >= limit:
                    break
            
            except Exception as e:
                print(f"Error processing line: {line}, Error: {str(e)}")
                continue

    consumer.close()
    return all_data


def save_data_to_csv(data, output_path="data/extracted_ratings.csv"):
    if not data:
        print("No data available to save.")
        return
    
    fieldnames = ['user_time', 'user_id', 'movie_id', 'movie_title', 'year', 'rating']
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w', newline='') as outfile:
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)
    
    print(f"Data successfully saved to {output_path}")


if __name__ == "__main__":
    latest_data = consume_kafka_logs()
    save_data_to_csv(latest_data, output_path="data/extracted_ratings.csv")
