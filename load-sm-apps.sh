# Load My Applications

echo "Remove sm-network and recreate"
docker network rm sm-network
docker network create sm-network
echo "sm-network created"

cd ~/sarah-maas-vault
echo "Start Custom Vault"
./docker-run-vault-nossl.sh local
echo "Unseal Vault"
./unseal-vault.sh
echo "Vault Ready"

cd ~/Young-Adult-ChatBot/kafka
echo "Start Kafka Server & UI"
./docker-run-kafka-local.sh local
echo "Kafka Server & UI Ready"
