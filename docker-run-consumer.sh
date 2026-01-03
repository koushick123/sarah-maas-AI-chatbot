docker container stop book-events-consumer
docker container rm -f book-events-consumer
if [ "$1" != "local" ]; then
   echo "Running with codespaces configuration"
   docker run -d --name book-events-consumer --network sm-network -v /workspaces/sarah-maas-AI-chatbot/pages_for_OCR:/pages_for_OCR koushick123/book-events-consumer:1.0
else
   echo "Running with local configuration"
   docker run -d --name book-events-consumer --network sm-network -v /home/koushick/Young-Adult-ChatBot/pages_for_OCR:/pages_for_OCR koushick123/book-events-consumer:1.0
fi