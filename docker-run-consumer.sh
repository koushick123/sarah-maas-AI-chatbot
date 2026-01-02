docker container stop book-events-consumer-container
docker container rm -f book-events-consumer-container
docker run -d --name book-events-consumer-container --network sm-network koushick123/book-events-consumer:1.0
