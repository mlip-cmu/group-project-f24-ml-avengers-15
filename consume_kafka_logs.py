from kafka import KafkaConsumer, TopicPartition
import re
import csv
from datetime import datetime
import os


def consume_last_kafka_logs(limit=200000):
    TOPIC_NAME = "movielog15"
    rate_pattern = re.compile(r'^.*?,\d+,GET /rate/.*?=\d+$')

    consumer = KafkaConsumer(
        bootstrap_servers=['localhost:9092'],
        enable_auto_commit=False,  # Prevent auto-commit for precise control
        value_deserializer=lambda x: x.decode('utf-8'),
    )

    # Assign consumer to the specific topic and its partitions
    partitions = consumer.partitions_for_topic(TOPIC_NAME)
    if not partitions:
        raise Exception(f"No partitions found for topic: {TOPIC_NAME}")

    topic_partitions = [TopicPartition(TOPIC_NAME, p) for p in partitions]
    consumer.assign(topic_partitions)

    # Set the consumer to start from the last `limit` messages
    for partition in topic_partitions:
        end_offset = consumer.end_offsets([partition])[partition]
        start_offset = max(0, end_offset - limit)
        consumer.seek(partition, start_offset)

    message_count = 0
    all_data = []

    try:
        # Process messages
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

                    entry = {
                        'user_time': time_str,
                        'user_id': userid,
                        'movie_id': movieid,
                        'movie_title': movietitle,
                        'year': year,
                        'rating': rating
                    }

                    all_data.append(entry)

                    # Print each fetched and processed entry
                    print(f"Fetched Entry #{message_count + 1}: {entry}")

                    message_count += 1

                    if message_count >= limit:
                        break

                except Exception as e:
                    print(f"Error processing line: {line}, Error: {str(e)}")
                    continue

    except KeyboardInterrupt:
        print("\nGracefully shutting down. Saving fetched data...")

    finally:
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
    try:
        latest_data = consume_last_kafka_logs(limit=500000)
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        # Save data even if interrupted
        save_data_to_csv(latest_data, output_path="data/extracted_ratings.csv")
