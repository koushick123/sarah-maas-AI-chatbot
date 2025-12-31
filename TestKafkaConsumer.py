import multiprocessing
from confluent_kafka import Consumer, KafkaError
from SarahMaasAzureOCR import read_text_from_cropped_ocr_image
from tinydb import TinyDB

def run_consumer(worker_id):
    db = TinyDB('map_of_page_nos_chapter_heading.json')
    
    # Configuration for 2025 KRaft and High-Latency OCR
    conf = {
        'bootstrap.servers': 'localhost:9094',
        'group.id': 'test-group-2025',
        'auto.offset.reset': 'earliest',
        'enable.auto.commit': True,
        'partition.assignment.strategy': 'roundrobin',
        
        # 1. Session Timeout: Detects network/crashes
        'session.timeout.ms': 45000,
        'heartbeat.interval.ms': 15000, # 1/3 of session timeout
        
        # 2. Max Poll Interval: Time allowed for OCR processing
        # Increase this if your OCR takes more than 5 minutes per batch
        'max.poll.interval.ms': 60000, # 1 minutes
        
        'error_cb': lambda err: print(f"Worker {worker_id} Error: {err}")
    }

    c = Consumer(conf)
    c.subscribe(['mytopic'])
    print(f"Worker {worker_id} started...")

    try:
        while True:
            msg = c.poll(5.0)
            if msg is None: continue
            if msg.error():
                if msg.error().code() != KafkaError._PARTITION_EOF:
                    print(f"Worker {worker_id} Error: {msg.error()}")
                continue

             # 1. Decode the bytes to a string
            raw_payload = msg.value().decode('utf-8')
            
            # 2. Parse the string into a Python dictionary
            # This will fail if the message was sent with single quotes
            import json
            val = json.loads(raw_payload) 
            
            print(f"Processing page: {val['page_num']}")
            
            # 3. Use the parsed dictionary
            image_text = read_text_from_cropped_ocr_image(val['image_path'], val['page_num'])
            db.insert({'page_num': val['page_num'], 'extracted_text': image_text[:10]})

    finally:
        c.close()

if __name__ == "__main__":
    # 3. Concurrent Consumers
    # Set this to match your topic partition count (e.g., 4)
    num_consumers = 10 
    processes = []

    for i in range(num_consumers):
        p = multiprocessing.Process(target=run_consumer, args=(i,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
