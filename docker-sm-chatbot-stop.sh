#!/bin/bash

# Stop and remove any existing container with the same name
echo "Stopping and removing existing container if it exists..."
docker container stop sarah-maas-research-assistant-container
docker container rm -f sarah-maas-research-assistant-container