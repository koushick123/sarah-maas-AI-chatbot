from kafka import KafkaProducer
import json

# Create producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9094'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = 'mytopic'

from tinydb import TinyDB, Query
db = TinyDB('map_of_page_nos_chapter_heading.json')

def sendKafkaMessage():
    # Send messages
    try:
        ChapterMetaData = Query()
        page_index = 21
        print("Starting to send messages...")
        while page_index < 38:
            docs = db.contains(ChapterMetaData.page_num == page_index)
            print(docs)
            # print(f"Checking page {ChapterMetaData.page_num} in database...")
            # if db.contains(ChapterMetaData.page_num == page_index):
            #     print(f"Page {page_index} found in database, skipping...")
            #     page_index += 1
            #     continue  # Skip pages already in the database
            # test_image_path = f"pages_for_OCR/page_empire-of-storms-20251128095023-0d555bf7_{page_index}.jpg"
            # print(f"Sending message for page {page_index} with image path {test_image_path}")
            # producer.send(topic, {"page_num": page_index, "image_path": test_image_path})
            # print(f"Sent message for page {page_index}")
            page_index += 1
        
    except Exception as e:
        print(f"Error: {str(e)}")

    finally:
        producer.close()

if __name__ == "__main__":
    sendKafkaMessage()