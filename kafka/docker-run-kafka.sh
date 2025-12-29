#!/bin/bash

docker-compose -f kafka-config-local.yaml down

if [ "$1" = "local" ]; then
    echo "Run Kafka with local configuration"
    docker-compose -f kafka-config-local.yaml up -d
else
    echo "Run Kafka with codespaces configuration"
    docker-compose -f kafka-config.yaml up -d
fi

echo "Kafka Container started"
