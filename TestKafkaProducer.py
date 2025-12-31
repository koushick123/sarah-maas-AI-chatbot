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
        for i in range(15):
            message = {
                'id': i,
                'message': f'Hello from NEW OFFSET message {i}',
                'timestamp': time.time()
            }

            # Send to Kafka
            future = producer.send(topic, value=message)

            # Block until message is sent (optional)
            result = future.get(timeout=10)

            print(f"Sent: {message}")

    except Exception as e:
        print(f"Error: {str(e)}")

    finally:
        producer.close()

if __name__ == "__main__":
    sendKafkaMessage()