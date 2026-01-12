#!/bin/bash

if [ "$1" = "local" ]; then
    echo "Run Kafka from local path"
    cd /home/koushick/Young-Adult-Chatbot/sarah-maas-AI-chatbot/kafka
else
    echo "Run Kafka from codespaces path"
    cd /workspaces/sarah-maas-AI-chatbot/kafka
fi

docker-compose -f kafka-config.yaml down
docker-compose -f kafka-config.yaml up -d

echo "Kafka Container started"
