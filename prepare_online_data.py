from kafka import KafkaConsumer
import re
from datetime import datetime, timedelta

TOPIC_NAME = "movielog15"
rate_pattern = re.compile(r'^.*?,\d+,GET /rate/.*?=\d+$')
end_time = datetime.now() - timedelta(days=10)
start_time = end_time - timedelta(days=1)


consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',  # Start reading from the beginning
    enable_auto_commit=True,
    group_id='rating-group',
    value_deserializer=lambda x: x.decode('utf-8')
)
# Process and store logs
total_count = 0
with open('extracted_ratings_sample.txt', 'w') as output_file:
    for message in consumer:
        line = message.value

        # Extract the timestamp from the ConsumerRecord
        message_timestamp = message.timestamp
        message_time = datetime.fromtimestamp(message_timestamp / 1000.0)

        # Check if the message timestamp is within the desired range
        if message_time < start_time:
            print(f"Message too old.")
            continue

        if message_time > end_time:
            print(f"Message too new.")
            break

        # Write matching logs to the file
        if rate_pattern.match(line):
            total_count += 1
            output_file.write(line + '\n')
            print(f"Matched rating log: {line}")
        
        if total_count == 150000:
            break
print(f"Total logs processed: {total_count}")
print("Extracted ratings saved to 'extracted_ratings.txt'")