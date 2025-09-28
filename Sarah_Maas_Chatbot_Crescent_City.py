from fastapi import FastAPI
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from pydantic import BaseModel
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from cryptography.fernet import Fernet
import os
from pymongo import MongoClient
import urllib.parse
from bson import ObjectId
import requests

# Initialize FastAPI app
app = FastAPI()

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

VAULT_URL = os.getenv("VAULT_ADDR", "127.0.0.1:8200")  # default to localhost
VAULT_RETRIEVER_URL = os.getenv("VAULT_RETRIEVER_URL", "127.0.0.1:8300")

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
        url = f"http://{VAULT_RETRIEVER_URL}/fetchVaultToken"
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
    url = f"https://{VAULT_URL}/v1/sm-secrets/data/openapi_mongodb_credentials"
    vault_token = fetch_vault_token()
    if vault_token.startswith("Error:") or vault_token.startswith("Request failed:"):
        raise ValueError(vault_token)
    headers = {
        "accept": "application/json",
        "X-Vault-Token": vault_token
    }
    cert_path = "vault-droplet/ssl/ca.crt"
    response = requests.get(url, headers=headers, verify=cert_path)
    response.raise_for_status()
    json_data = response.json()
    key_value = json_data["data"]["data"].get(key)
    print(f"Fetched value for {key}: {key_value is not None}")
    return key_value

username = urllib.parse.quote_plus(decrypt_mongo_user())
password = urllib.parse.quote_plus(decrypt_mongo_password())
host_url = decrypt_mongo_hosturl()
uri = f"mongodb+srv://{username}:{password}@{host_url}/?retryWrites=true&w=majority&appName=dev-cluster"

client = MongoClient(uri)
db = client["sarah-maas-db"]
collection_books = db["sarah-maas-books"]
collection_books_summaries = db["sarah-maas-books-summaries"]

UI_ORIGIN_URL = os.getenv("UI_ORIGIN_URL","localhost:4200")

# Allow specific origins (replace with your Angular dev server URL)
origins = [
    f"http://localhost:4200"  # Angular local dev
    #"https://your-angular-app.com"  # Prod Angular app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # List of allowed origins
    allow_credentials=True,           # Allow cookies/auth headers
    allow_methods=["*"],
    allow_headers=["*"],               # Allow all headers
)

class Chapter(BaseModel):
    book_name: str
    part: str
    chapter_name: str
    chapter_content: str
    summary_option: str

class ChapterSummary(BaseModel):
    book_name: str
    part: str
    chapter_name: str
    chapter_summary: str
    summary_option: str
    doc_id: str


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

os.environ["OPENAI_API_KEY"] = decrypt_openapi_key()  # Decrypt the OpenAI API key

# API for health check
@app.get("/healthcheck")
def healthcheck():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok", "message": "Sarah Maas AI Chatbot is running!"}


# "Summary 1 - Summarize entire chapter using regular ChatGPT",
# "Summary 3 - Merge Summary 1 and Summary 2 using regular ChatGPT"
def summarize_with_gpt4turbo(context_chapter_summary, option):
    system_message = (
        "You are a knowledgeable literary research assistant with deep familiarity "
        "with Sarah J. Maas's *Crescent City* series of books.\n"
        "Focus on offering thoughtful, research-level insight into the text.\n"
        "Avoid generic filler and do not add introductory phrases like 'In this chapter of Crescent City' or anything similar.\n"
    )

    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        HumanMessagePromptTemplate.from_template(
            "Answer the following question using ONLY the context provided.\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    ])

    selected_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1)

    chain = LLMChain(llm=selected_llm, prompt=chat_prompt)
    if option == "summary1" or option == "summary2":
        question="Summarize the chapter in detail, focusing on characterisation and plot progression"
    else:
        question="Merge the two summaries into a single coherent summary, focusing on characterisation and plot progression"

    return chain.run(context=context_chapter_summary, question=question).strip()

# "Summary 2 - Summarize chapter part by part and merge"
def summarize_with_langchain(chapter_context: str) -> str:
    # Use LangChain's summarization chain
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1)
    docs = [Document(page_content=chapter_context)]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    result = chain.run(docs)
    return result.strip()

@app.get("/logs")
def read_log_file():
    """
    Read the specified log file and return all lines as a list of strings.
    """
    try:
        with open("logfile.txt", "r") as f:
            lines = f.readlines()
        html_lines = "This is for logs<br>Line 1:<br>"
        for line in lines:
            html_lines += line.replace("\n", "<br>")
        return Response(content=html_lines.strip(), media_type="text/html")
    except FileNotFoundError:
        return {"error": f"Log file logfile.txt not found."}
    except Exception as e:
        return {"error": str(e)}

@app.post("/logs/clean")
def clean_log_file():
    """
    Truncate logfile.txt, removing all contents.
    """
    try:
        open("logfile.txt", "w").close()
        return {"message": "Log file cleaned successfully."}
    except Exception as e:
        return {"error": str(e)}

# API to fetch the chapter contents based on book and chapter selection
@app.get("/book/{book_name}/chapter/{chapter_name}/contents")
def fetch_book_contents(book_name : str, chapter_name: str):
    print("Book Name:", book_name)
    print("Chapter Name:", chapter_name)
    # Get the chapter contents as per selection
    chapter_name = chapter_name.replace("-"," ")
    if book_name != "Select a Book" and chapter_name != "Select a Chapter":
        if book_name == "Crescent-City-Book-1":
            chapter = collection_books.find({"Name": chapter_name})
            chapter_content = chapter.next().get("Page Content", "No content found for this chapter.")
            if int(chapter_name[chapter_name.index(" "):]) > 9:
                return chapter_content[2:]
            else:
                return chapter_content

# API to fetch the chapter and part titles based on book selection
@app.get("/book/{book_name}/chapters")
def fetch_chapter_titles(book_name: str):
    """
    Fetch chapter titles based on the selected book.
    """
    if book_name != "Select a Book":
        if book_name == "Crescent-City-Book-1":
            chapter_names = [doc["Name"] for doc in collection_books.find({}, {"Name": 1, "_id": 0})]
            part_chapter_map = {}
            for chapter_name in chapter_names:
                chapter_docs = collection_books.find({"Name": chapter_name.replace("'", "")})
                doc = chapter_docs.next()
                part = doc.get("Part")
                name = doc.get("Name")
                if part not in part_chapter_map:
                    part_chapter_map[part] = []
                part_chapter_map[part].append(name)
            return part_chapter_map
    return {"error": "No chapters found for the selected book."}

# API to save the chapter summary
@app.post("/chapter/save")
def save_chapter_summary(chapter_summary: ChapterSummary):
    new_data = {
        "Name": chapter_summary.chapter_name.replace("-", " "),
        "Part": chapter_summary.part,
        "Summary Option": chapter_summary.summary_option,
        "Book Name": chapter_summary.book_name,
        "Summary": chapter_summary.chapter_summary
    }
    if chapter_summary.doc_id:
        object_id = ObjectId(chapter_summary.doc_id)
        collection_books_summaries.update_one({"_id": object_id}, {"$set": new_data}, upsert=True)
        doc_id = chapter_summary.doc_id
    else:
        result = collection_books_summaries.insert_one(new_data)
        doc_id = result.inserted_id
    return {"message": "Chapter summary saved successfully.", "doc_id": str(doc_id)}

#API get fetch saved chapter summaries
@app.post("/chapter/summaries")
def fetch_chapter_summaries(chapterFromUI: ChapterSummary):
    """
    Fetch all saved chapter summaries.
    """
    # Fetch chapter summary from MongoDB
    query = {
        "Name": chapterFromUI.chapter_name.replace("-", " "),
        "Book Name": chapterFromUI.book_name,
        "Part": chapterFromUI.part,
        "Summary Option": chapterFromUI.summary_option
    }
    summary_doc = collection_books_summaries.find_one(query)
    if summary_doc:
        return {"summary": summary_doc.get("Summary", ""), "doc_id": str(summary_doc.get("_id"))}
    else:
        return {"summary": "No chapter summaries found.", "doc_id": "-1"}

# API to generate chapter summary based on selected option
@app.post("/chapter/summary")
def generate_chapter_summary(chapter_to_summarize: Chapter):
    """
    Generate a summary for the given chapter context based on the selected summary option.
    """
    if chapter_to_summarize.book_name == "Crescent-City-Book-1":
        page_content = chapter_to_summarize.chapter_content

        if chapter_to_summarize.summary_option == "summary1":
            return summarize_with_gpt4turbo(page_content, chapter_to_summarize.summary_option)
        elif chapter_to_summarize.summary_option == "summary2":
            return summarize_with_langchain(page_content)
        elif chapter_to_summarize.summary_option == "summary3":
            return summarize_with_gpt4turbo(page_content, chapter_to_summarize.summary_option)
        else:
            return {"error": "Invalid summary option selected."}

    else:
        return {"error": "Selected book is not available."}

