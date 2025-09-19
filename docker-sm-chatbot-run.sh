#!/bin/bash

# Stop and remove any existing container with the same name
echo "Stopping and removing existing container if it exists..."
docker container stop sarah-maas-research-assistant-container
docker container rm -f sarah-maas-research-assistant-container

# Remove the existing image if it exists
echo "Removing existing image if it exists..."
docker image rm -f koushick123/sarah-maas-research-assistant:1.0

# Build the Docker image with the given tag
echo "Building the Docker image..."
docker build -t koushick123/sarah-maas-research-assistant:1.0 .

# Run the Docker container
echo "Running the Docker container..."
docker run -d --name sarah-maas-research-assistant-container -p 8000:8000 koushick123/sarah-maas-research-assistant:1.0
