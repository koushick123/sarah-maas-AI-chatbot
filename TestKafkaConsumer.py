from confluent_kafka import Consumer, KafkaError

def error_cb(err):
    print(f"Global Error: {err}")

c = Consumer({
    'bootstrap.servers': 'localhost:9094',
    'group.id': 'test-group-2025',
    'auto.offset.reset': 'earliest',
    'session.timeout.ms': 45000,      # Give KRaft more time to coordinate
    'error_cb': error_cb,            # This will tell you IF it can't connect
    'enable.auto.commit': True,
    'partition.assignment.strategy': 'roundrobin'
})

c.subscribe(['mytopic'])

print("Starting consumer...")
try:
    while True:
        msg = c.poll(1.0) # Check every 1 second
        if msg is None:
            continue
        if msg.error():
            if msg.error().code() == KafkaError._PARTITION_EOF:
                continue
            else:
                print(f"Error: {msg.error()}")
        else:
            print(f"Success! Received: {msg.value().decode('utf-8')} Offset: {msg.offset()} Partition: {msg.partition()}")
finally:
    c.close()
