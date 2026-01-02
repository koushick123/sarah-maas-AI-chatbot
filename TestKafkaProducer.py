from kafka import KafkaProducer
import json
from pymongo import MongoClient
import urllib.parse
from SarahMaasChatbotCrescentCity import decrypt_mongo_user, decrypt_mongo_password, decrypt_mongo_hosturl

username = urllib.parse.quote_plus(decrypt_mongo_user())
password = urllib.parse.quote_plus(decrypt_mongo_password())
host_url = decrypt_mongo_hosturl()
uri = f"mongodb+srv://{username}:{password}@{host_url}/?retryWrites=true&w=majority&appName=dev-cluster"

client = MongoClient(uri)
db = client["sarah-maas-db"]
sm_map_page_nos_chap_heading_collection = db['sarah-maas-map-page-nos-chapter-heading']

# Create producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9094'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = 'mytopic'

def sendKafkaMessage():
    # Send messages
    try:
        page_index = 21
        print("Starting to send messages...")
        while page_index < 122:
            if sm_map_page_nos_chap_heading_collection.find_one({"page_num": page_index}):
                print(f"Page {page_index} found in database, skipping...")
                page_index += 1
                continue  # Skip pages already in the database
            test_image_path = f"pages_for_OCR/page_empire-of-storms-20251128095023-0d555bf7_{page_index}.jpg"
            print(f"Sending message for page {page_index} with image path {test_image_path}")
            producer.send(topic, {"page_num": page_index, "image_path": test_image_path})
            page_index += 1
        
    except Exception as e:
        print(f"Error: {str(e)}")

    finally:
        producer.close()

if __name__ == "__main__":
    sendKafkaMessage()