from kafka import KafkaConsumer
import json

# Create consumer
consumer = KafkaConsumer(
    'topic1',
    bootstrap_servers=['localhost:9092'],
    auto_offset_reset='earliest',  # Start from beginning
    enable_auto_commit=True,
    group_id='my-consumer-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

print("Waiting for messages...")

def consumeMessages():
    try:
        for message in consumer:
            print(f"Received: {message.value}")
            print(f"Partition: {message.partition}, Offset: {message.offset}")
            print("---")

    except KeyboardInterrupt:
        print("Stopped by user")

    finally:
        consumer.close()

if __name__ == "__main__":
    consumeMessages()