from kafka import KafkaConsumer
import re
from datetime import datetime, timedelta
import csv

def consume_kafka_logs(output_file_path):
    TOPIC_NAME = "movielog15"
    rate_pattern = re.compile(r'^.*?,\d+,GET /rate/.*?=\d+$')
    end_time = datetime.now()
    start_time = end_time - timedelta(days=1)

    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',  # Start reading from the beginning
        enable_auto_commit=True,
        group_id='rating-group',
        value_deserializer=lambda x: x.decode('utf-8')
    )

    total_count = 0
    with open(output_file_path, 'w') as output_file:
        for message in consumer:
            line = message.value

            message_timestamp = message.timestamp
            message_time = datetime.fromtimestamp(message_timestamp / 1000.0)

            if message_time < start_time:
                continue

            if message_time > end_time:
                break

            if rate_pattern.match(line):
                total_count += 1
                output_file.write(line + '\n')



def convert_ratings_txt_to_csv(input_path, output_path):
    with open(input_path, 'r') as infile, open(output_path, 'w', newline='') as outfile:
        fieldnames = ['user_time', 'user_id', 'movie_id', 'movie_title', 'year', 'rating']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        
        writer.writeheader()
        
        for line in infile:
            parts = line.strip().split(',')
            
            if len(parts) < 3:
                continue
            
            try:
                time = parts[0]
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
                
                writer.writerow({
                    'user_time': time,
                    'user_id': userid,
                    'movie_id': movieid,
                    'movie_title': movietitle,
                    'year': year,
                    'rating': rating
                })
            
            except Exception as e:
                print(f"Error processing line: {line}, Error: {str(e)}")
                continue
