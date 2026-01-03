import multiprocessing
from confluent_kafka import Consumer, KafkaError
from SarahMaasAzureOCR import read_text_from_cropped_ocr_image
from pymongo import MongoClient
import urllib.parse
from DecryptCredentials import decrypt_mongo_user, decrypt_mongo_password, decrypt_mongo_hosturl

username = urllib.parse.quote_plus(decrypt_mongo_user())
password = urllib.parse.quote_plus(decrypt_mongo_password())
host_url = decrypt_mongo_hosturl()
uri = f"mongodb+srv://{username}:{password}@{host_url}/?retryWrites=true&w=majority&appName=dev-cluster"

# Need to create a new MongoClient in each process for thread safety
def createMongoClient():
    client = MongoClient(uri)
    db = client["sarah-maas-db"]
    return db['sarah-maas-map-page-nos-chapter-heading']

def book_events_consumer(worker_id):

    try: 
        # Configuration for 2025 KRaft and High-Latency OCR
        conf = {
            'bootstrap.servers': 'kafka-local:9094',
            'group.id': 'test-group-2025',
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': True,
            'partition.assignment.strategy': 'roundrobin',
            
            # 1. Session Timeout: Detects network/crashes
            'session.timeout.ms': 45000,
            'heartbeat.interval.ms': 15000, # 1/3 of session timeout
            
            # 2. Max Poll Interval: Time allowed for OCR processing
            # Increase this if your OCR takes more than 5 minutes per batch
            'max.poll.interval.ms': 300000, # 5 minutes
            
            'error_cb': lambda err: print(f"Worker {worker_id} Error: {err}")
        }

        c = Consumer(conf)
        c.subscribe(['mytopic'])
        print(f"Worker {worker_id} started...")
    
        while True:
            msg = c.poll(5.0)
            if msg is None: continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"Worker {worker_id} Error: {msg.error()}")
                continue

             # 1. Decode the bytes to a string
            raw_payload = msg.value().decode('utf-8')
            print(f"Worker {worker_id} received message: {raw_payload} from partition: {msg.partition()} at offset: {msg.offset()}")
            
            # 2. Parse the string into a Python dictionary
            # This will fail if the message was sent with single quotes
            import json
            val = json.loads(raw_payload) 
            
            print(f"Processing page: {val['page_num']} for image path {'/'+val['image_path']}")
            
            # 3. Use the parsed dictionary
            image_text = read_text_from_cropped_ocr_image('/'+val['image_path'], val['page_num'])
            print(f"image text for page {val['page_num']}: {image_text[:5]}...")
            createMongoClient().insert_one({"book_id": val['book_id'], "page_num": val['page_num'], "extracted_text": image_text[:15]})

    except Exception as e:
        print(f"Worker {worker_id} Exception: {str(e)}")
    finally:
        c.close()

if __name__ == "__main__":
    print("Starting multiple Book Events Consumers...")
    num_consumers = 10 
    processes = []

    for i in range(num_consumers):
        p = multiprocessing.Process(target=book_events_consumer, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
