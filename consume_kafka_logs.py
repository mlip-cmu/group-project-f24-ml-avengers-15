from kafka import KafkaConsumer
import re
import csv
import pandas as pd
import glob
from datetime import datetime, timedelta



def consume_kafka_logs():
    TOPIC_NAME = "movielog15"
    rate_pattern = re.compile(r'^.*?,\d+,GET /rate/.*?=\d+$')

    consumer = KafkaConsumer(
        TOPIC_NAME,
        bootstrap_servers=['localhost:9092'],
        auto_offset_reset='earliest',
        enable_auto_commit=True,
        group_id='rating-group',
        value_deserializer=lambda x: x.decode('utf-8')
    )

    for message in consumer:
        line = message.value

        if rate_pattern.match(line):
            message_timestamp = message.timestamp
            message_time = datetime.fromtimestamp(message_timestamp / 1000.0)
            date_str = message_time.strftime('%Y-%m-%d')
            output_file_path = f"data/logs_{date_str}.txt"

            with open(output_file_path, 'a') as output_file:
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


def load_recent_data(days=3):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    data_frames = []

    for i in range(days):
        date_str = (start_date + timedelta(days=i)).strftime('%Y-%m-%d')
        csv_file = f"data/logs_{date_str}.csv"
        txt_file = f"data/logs_{date_str}.txt"
        
        # Convert TXT to CSV if CSV doesn't exist
        if not os.path.exists(csv_file) and os.path.exists(txt_file):
            convert_ratings_txt_to_csv(txt_file, csv_file)
        
        # Read the CSV file
        if os.path.exists(csv_file):
            df = pd.read_csv(csv_file)
            data_frames.append(df)

    if data_frames:
        return pd.concat(data_frames, ignore_index=True)
    else:
        print("No data available for retraining.")
        return pd.DataFrame()
    

if __name__ == "__main__":
    consume_kafka_logs()