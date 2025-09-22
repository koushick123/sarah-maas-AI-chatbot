#!/usr/bin/env bash
set -euo pipefail

# ============================
# Configurable variables
# ============================
VAULT_HOSTNAME="vault.local"
VAULT_IP="127.0.0.1"
SSL_DIR="/etc/vault/ssl"

# ============================
# Prepare directories
# ============================
WORKDIR="$(mktemp -d)"
echo "[INFO] Working directory: $WORKDIR"

# Create SSL_DIR if it does not exist
if [ ! -d "$SSL_DIR" ]; then
 echo "[INFO] SSL Directory $SSL_DIR does not exist. Creating..."
 sudo mkdir -p "$SSL_DIR"
 sudo chown root:root "$SSL_DIR"
 sudo chmod 755 "$SSL_DIR"
else
 echo "[INFO] Using existing SSL directory:$SSL_DIR"
fi

# ============================
# 1. Create a CA
# ============================
echo "[INFO] Generating CA..."
openssl genrsa -out "$WORKDIR/ca.key" 4096
openssl req -x509 -new -nodes -key "$WORKDIR/ca.key" \
  -sha256 -days 3650 \
  -subj "/C=IN/ST=Karnataka/L=Bengaluru/O=MyOrg/OU=Vault CA/CN=Vault-Local-CA" \
  -out "$WORKDIR/ca.crt"

# ============================
# 2. Create server cert
# ============================
echo "[INFO] Generating Vault server certificate..."

cat > "$WORKDIR/server.cnf" <<EOF
[req]
default_bits = 4096
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = req_ext

[dn]
C=IN
ST=Karnataka
L=Bengaluru
O=MyOrg
OU=Vault Servers
CN = ${VAULT_HOSTNAME}

[req_ext]
subjectAltName = @alt_names

[alt_names]
DNS.1 = ${VAULT_HOSTNAME}
DNS.2 = localhost
IP.1 = ${VAULT_IP}
EOF

openssl genrsa -out "$WORKDIR/vault-server.key" 4096
openssl req -new -key "$WORKDIR/vault-server.key" -out "$WORKDIR/vault-server.csr" -config "$WORKDIR/server.cnf"
openssl x509 -req -in "$WORKDIR/vault-server.csr" \
  -CA "$WORKDIR/ca.crt" -CAkey "$WORKDIR/ca.key" -CAcreateserial \
  -out "$WORKDIR/vault-server.crt" -days 825 -sha256 \
  -extfile "$WORKDIR/server.cnf" -extensions req_ext

# ============================
# 3. Create client cert (PEM)
# ============================
echo "[INFO] Generating client certificate..."

cat > "$WORKDIR/client.cnf" <<EOF
[ req ]
default_bits = 4096
prompt = no
default_md = sha256
distinguished_name = dn
req_extensions = req_ext

[ dn ]
C=IN
ST=Karnataka
L=Bengaluru
O=MyOrg
OU=Clients
CN=client1

[ req_ext ]
extendedKeyUsage = clientAuth
EOF

openssl genrsa -out "$WORKDIR/client1.key" 4096
openssl req -new -key "$WORKDIR/client1.key" -out "$WORKDIR/client1.csr" -config "$WORKDIR/client.cnf"
openssl x509 -req -in "$WORKDIR/client1.csr" \
  -CA "$WORKDIR/ca.crt" -CAkey "$WORKDIR/ca.key" -CAcreateserial \
  -out "$WORKDIR/client1.crt" -days 365 -sha256 \
  -extfile "$WORKDIR/client.cnf" -extensions req_ext

# ============================
# 4. Copy certs to /etc/vault/ssl
# ============================
echo "[INFO] Installing certificates to $SSL_DIR ..."

sudo cp "$WORKDIR/ca.crt" "$SSL_DIR/"
sudo cp "$WORKDIR/vault-server.crt" "$WORKDIR/vault-server.key" "$SSL_DIR/"
sudo cp "$WORKDIR/client1.crt" "$WORKDIR/client1.key" "$SSL_DIR/"

# set permissions (adjust vault:vault if your service runs as another user)
sudo chown -R koushick:koushick "$SSL_DIR"
sudo chmod 640 "$SSL_DIR/vault-server.key" "$SSL_DIR/client1.key"
sudo chmod 644 "$SSL_DIR/"*.crt

# ============================
# Done
# ============================
echo "[INFO] Certificates generated and installed in $SSL_DIR"
echo " - CA:        $SSL_DIR/ca.crt"
echo " - Server:    $SSL_DIR/vault-server.crt , $SSL_DIR/vault-server.key"
echo " - Client:    $SSL_DIR/client1.crt , $SSL_DIR/client1.key"
echo
echo "Next steps:"
echo " 1. Update your vault.hcl listener block to use:"
echo "      tls_cert_file = \"$SSL_DIR/vault-server.crt\""
echo "      tls_key_file  = \"$SSL_DIR/vault-server.key\""
echo "      tls_client_ca_file = \"$SSL_DIR/ca.crt\"   # optional for mTLS"
echo " 2. Restart Vault: sudo systemctl restart vault"
echo " 3. Test with curl:"
echo "      curl --cacert $SSL_DIR/ca.crt https://${VAULT_IP}:8200/v1/sys/health"
echo "      curl --cacert $SSL_DIR/ca.crt --cert $SSL_DIR/client1.crt --key $SSL_DIR/client1.key https://${VAULT_IP}:8200/v1/sys/health"
