storage "file" {
  path = "vault-data"
}

# TCP listener with TLS enabled
listener "tcp" {
  address       = "0.0.0.0:8200"
  tls_cert_file = "vault-droplet/ssl/vault-server.crt"
  tls_key_file  = "vault-droplet/ssl/vault-server.key"

  # Require clients to present a certificate signed by your CA
  # Comment this out if you want normal TLS (server-only auth)
  tls_client_ca_file = "vault-droplet/ssl/ca.crt"
}

# This is the address Vault advertises to clients and in redirects
api_addr = "https://64.227.147.196:8200"

# Enable UI
ui = true

