import requests
import os
from cryptography.fernet import Fernet

SSL_FLAG = os.getenv("SSL_FLAG", "false")
# Update VAUTL_ADDR as per Prod if SSL_FLAG is true
VAULT_ADDR = "<prod-url>"
VAULT_RETRIEVER_ADDR = os.getenv("VAULT_RETRIEVER_ADDR")

def decrypt_mongo_password():
    """
    Function to decrypt MongoDB password.
    """
    encryptionkey = fetch_decryption_key_from_vault("FERNET_KEY_MONGO_PASSWORD")  # Fetch the encryption key from Vault
    if not encryptionkey:
        raise ValueError("FERNET_KEY_MONGO_PASSWORD not set")
    fernet = Fernet(encryptionkey)
    with open("encryptedmongopassword.txt") as file:
        encrypted_mongo_password = file.read().encode()
    return fernet.decrypt(encrypted_mongo_password).decode()


def decrypt_mongo_user():
    """
    Function to decrypt MongoDB username.
    """
    encryptionkey = fetch_decryption_key_from_vault("FERNET_KEY_MONGO_USERNAME")  # Fetch the encryption key from Vault
    if not encryptionkey:
        raise ValueError("FERNET_KEY_MONGO_USERNAME not set")
    fernet = Fernet(encryptionkey)
    with open("encryptedmongouser.txt") as file:
        encrypted_mongo_password = file.read().encode()
    return fernet.decrypt(encrypted_mongo_password).decode()


def decrypt_mongo_hosturl():
    """
    Function to decrypt MongoDB host URL.
    """
    encryptionkey = fetch_decryption_key_from_vault("FERNET_KEY_MONGO_HOSTURL")  # Fetch the encryption key from Vault
    if not encryptionkey:
        raise ValueError("FERNET_KEY_MONGO_HOSTURL not set")
    fernet = Fernet(encryptionkey)
    with open("encryptedmongohosturl.txt") as file:
        encrypted_mongo_hosturl = file.read().encode()
    return fernet.decrypt(encrypted_mongo_hosturl).decode()


def decrypt_azure_ocr_api():
    """
    Function to decrypt Azure OCR API.
    """
    encryptionkey = fetch_decryption_key_from_vault("FERNET_KEY_AZURE_OCR_KEY")  # Fetch the encryption key from Vault
    if not encryptionkey:
        raise ValueError("FERNET_KEY_AZURE_OCR_KEY not set")
    fernet = Fernet(encryptionkey)
    with open("encryptedazureocrapi.txt") as file:
        encrypted_azure_ocr_api = file.read().encode()
    return fernet.decrypt(encrypted_azure_ocr_api).decode()


def decrypt_azure_ocr_host():
    """
    Function to decrypt Azure OCR Host.
    """
    encryptionkey = fetch_decryption_key_from_vault("FERNET_KEY_AZURE_OCR_HOST")  # Fetch the encryption key from Vault
    if not encryptionkey:
        raise ValueError("FERNET_KEY_AZURE_OCR_HOST not set")
    fernet = Fernet(encryptionkey)
    with open("encryptedazureocrhost.txt") as file:
        encrypted_azure_ocr_api = file.read().encode()
    return fernet.decrypt(encrypted_azure_ocr_api).decode()

def fetch_vault_token() -> str:
    """
    Fetch Vault access token by retrieving VM metadata (vmId, publicKeys)
    and sending it to the Vault token retrieval service.

    Returns:
        str: Vault access token if success, or error message if unauthorized.
    """
    try:
        # Fetch VM metadata
        # Using IMDS metadata service for DigitalOcean. This IP address is non-routable and cannot be accessed externally.
        vm_id = requests.get("http://169.254.169.254/metadata/v1/id", timeout=5).text.strip()
        public_keys = requests.get("http://169.254.169.254/metadata/v1/public-keys", timeout=5).text.strip()

        # Vault token retrieval service
        url = f"http://{VAULT_RETRIEVER_ADDR}/fetchVaultToken"
        payload = {
            "vmId": vm_id,
            "publicKeys": public_keys
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=payload, headers=headers, timeout=10)
        response.raise_for_status()

        data = response.json()
        if data.get("result") == "success":
            return data.get("token")
        else:
            return f"Error: {data.get('error', 'Unauthorized VM for accessing Vault token')}"
    except requests.RequestException as e:
        return f"Request failed: {e}"


def fetch_decryption_key_from_vault(key: str) -> str:
    vault_token = fetch_vault_token()
    if vault_token.startswith("Error:") or vault_token.startswith("Request failed:"):
        raise ValueError(vault_token)
    headers = {
        "accept": "application/json",
        "X-Vault-Token": vault_token
    }
    cert_path = "vault-droplet/ssl/ca.crt"
    if SSL_FLAG == "true":
        url = f"https://{VAULT_ADDR}/v1/sm-secrets/data/openapi_mongodb_credentials"
        response = requests.get(url, headers=headers, verify=cert_path)
    else:
        url = f"http://my-vault-container-nossl:8300/v1/sm-secrets/data/openapi_mongodb_credentials"
        response = requests.get(url, headers=headers)
    response.raise_for_status()
    json_data = response.json()
    key_value = json_data["data"]["data"].get(key)
    print(f"Fetched value for {key}: {key_value is not None}")
    return key_value


def decrypt_openapi_key():
    """
    Function to decrypt OpenAPI key.
    """
    encryption_key = fetch_decryption_key_from_vault("FERNET_KEY")
    if not encryption_key:
        raise ValueError("FERNET_KEY not set")
    fernet = Fernet(encryption_key)
    with open("encryptedopenapi.txt") as file:
        encrypted_api = file.read().encode()
    return fernet.decrypt(encrypted_api).decode()

if __name__ == "__main__":
    print("MongoDB User:", decrypt_mongo_user())