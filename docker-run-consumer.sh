docker container stop book-events-consumer
docker container rm -f book-events-consumer
docker run -d --name book-events-consumer --network sm-network koushick123/book-events-consumer:1.0
