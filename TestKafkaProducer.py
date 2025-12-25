from kafka import KafkaProducer
import json
import time

# Create producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9094'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = 'mytopic'

def sendKafkaMessage():
    # Send messages
    try:
        for i in range(10):
            message = {
                'id': i,
                'message': f'Hello from message {i}',
                'timestamp': time.time()
            }

            # Send to Kafka
            future = producer.send(topic, value=message)

            # Block until message is sent (optional)
            result = future.get(timeout=10)

            print(f"Sent: {message}")
            time.sleep(1)

    except Exception as e:
        print(f"Error: {e}")

    finally:
        producer.close()

if __name__ == "__main__":
    sendKafkaMessage()