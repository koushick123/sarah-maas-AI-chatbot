storage "file" {
  path = "/home/koushick/Young-Adult-ChatBot/vault-data"
}

# TCP listener with TLS enabled
listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_cert_file = "/home/koushick/Young-Adult-ChatBot/vault/ssl/vault-server.crt"
  tls_key_file  = "/home/koushick/Young-Adult-ChatBot/vault/ssl/vault-server.key"

  # Require clients to present a certificate signed by your CA
  # Comment this out if you want normal TLS (server-only auth)
  tls_client_ca_file = "/home/koushick/Young-Adult-ChatBot/vault/ssl/ca.crt"
}

# This is the address Vault advertises to clients and in redirects
api_addr = "https://localhost:8200"

# Enable UI
ui = true
