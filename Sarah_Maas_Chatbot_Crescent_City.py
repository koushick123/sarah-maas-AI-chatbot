import os
import urllib.parse

import requests
from bson import ObjectId
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from gridfs import GridFS
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from pydantic import BaseModel
from pymongo import MongoClient
from strands import Agent, tool
from strands.models.openai import OpenAIModel

# Initialize FastAPI app
app = FastAPI()

UI_ORIGIN_URL = os.getenv("UI_ORIGIN_URL")
#VAULT_ADDR = os.getenv("VAULT_ADDR")
SSL_FLAG = os.getenv("SSL_FLAG", "false")
VAULT_ADDR = "verbose-space-guide-69pj5p75vrp3pp9-8300.app.github.dev"
VAULT_RETRIEVER_ADDR = os.getenv("VAULT_RETRIEVER_ADDR")

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

class BookStaging(BaseModel):
    book_name: str
    book_id: str
    pdf_file_id: str  # GridFS file ID
    file_size: int
    content_type: str = "application/pdf"

# Allow specific origins (replace with your Angular dev server URL)
origins = [
    f"http://{UI_ORIGIN_URL}"  # Angular local dev
    #"https://your-angular-app.com"  # Prod Angular app
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,            # List of allowed origins
    allow_credentials=True,           # Allow cookies/auth headers
    allow_methods=["*"],
    allow_headers=["*"],               # Allow all headers
)

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
    url = f"https://{VAULT_ADDR}/v1/sm-secrets/data/openapi_mongodb_credentials"
    vault_token = fetch_vault_token()
    if vault_token.startswith("Error:") or vault_token.startswith("Request failed:"):
        raise ValueError(vault_token)
    headers = {
        "accept": "application/json",
        "X-Vault-Token": vault_token
    }
    cert_path = "vault-droplet/ssl/ca.crt"
    if SSL_FLAG == "true":
        response = requests.get(url, headers=headers, verify=cert_path)
    else:
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

os.environ["OPENAI_API_KEY"] = decrypt_openapi_key()  # Decrypt the OpenAI API key

username = urllib.parse.quote_plus(decrypt_mongo_user())
password = urllib.parse.quote_plus(decrypt_mongo_password())
host_url = decrypt_mongo_hosturl()
uri = f"mongodb+srv://{username}:{password}@{host_url}/?retryWrites=true&w=majority&appName=dev-cluster"

client = MongoClient(uri)
db = client["sarah-maas-db"]
collection_books = db["sarah-maas-books"]
collection_books_summaries = db["sarah-maas-books-summaries"]
collection_book_staging = db["sarah-maas-books-staging"]

fs = GridFS(db)

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
    print("Using Langchain for summary", chapter_context)
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1)
    docs = [Document(page_content=chapter_context)]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    result = chain.run(docs)
    print("Langchain Summary Result:", result)
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
    if chapter_summary.doc_id != "-1":
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

import json
import fitz
from typing import AsyncGenerator, List
from fastapi.responses import StreamingResponse

def chunk_text(text: str, chunk_size: int = 25000, overlap: int = 500) -> List[str]:
    """
    Split text into chunks of specified size with overlap to preserve context.

    Args:
        text: The full text to chunk
        chunk_size: Maximum characters per chunk (default 25000 to leave room for instructions)
        overlap: Number of characters to overlap between chunks for context

    Returns:
        List of text chunks
    """
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size

        # If this is not the last chunk, try to break at a paragraph or sentence
        if end < len(text):
            # Look for paragraph break
            paragraph_break = text.rfind('\n\n', start, end)
            if paragraph_break != -1 and paragraph_break > start + chunk_size // 2:
                end = paragraph_break
            else:
                # Look for sentence break
                sentence_break = text.rfind('. ', start, end)
                if sentence_break != -1 and sentence_break > start + chunk_size // 2:
                    end = sentence_break + 1

        print("Appending chunk:", start, "to", end," with size ",len(text[start:end]))
        chunks.append(text[start:end])

        # Move start position, accounting for overlap
        start = end - overlap if end < len(text) else end

    return chunks


@tool
def get_book_chunks(book_id: str, chunk_size: int = 25000) -> dict:
    """
    Download PDF and return chunked text for analysis.
    This tool fetches a book PDF, extracts text, and chunks it for processing.

    Args:
        book_id: The unique identifier of the book to download
        chunk_size: Maximum characters per chunk

    Returns:
        Dictionary with chunks, metadata, and chunk info
    """
    import tempfile

    try:
        print(f"🔧 TOOL CALLED: get_book_chunks with book_id={book_id}")

        # Get staging document
        doc = collection_book_staging.find_one({"book_id": book_id})
        if not doc:
            return {"error": "Book not found", "success": False}

        # Retrieve PDF from GridFS
        file_id = ObjectId(doc["file_id"])
        pdf_file = fs.get(file_id)
        pdf_binary = pdf_file.read()

        print(f"📄 PDF fetched, size: {len(pdf_binary)} bytes")

        # Create a temporary file to write the PDF binary
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_pdf:
            temp_pdf.write(pdf_binary)
            temp_pdf_path = temp_pdf.name

        try:
            # Open PDF using file path with context manager
            with fitz.open(temp_pdf_path) as pdf_document:
                full_text = ""
                page_count = len(pdf_document)

                print(f"📖 Extracting text from {page_count} pages...")

                for page_num in range(page_count):
                    page = pdf_document[page_num]
                    full_text += page.get_text() + "\n"

            print(f"✅ Text extracted. Total characters: {len(full_text)}")

            # Chunk the text
            chunks = chunk_text(full_text, chunk_size)
            print(f"📦 Text divided into {len(chunks)} chunks")

            return {
                "chunks": chunks,
                "total_chunks": len(chunks),
                "page_count": page_count,
                "file_name": f"{book_id}.pdf",
                "file_id": book_id,
                "total_characters": len(full_text),
                "success": True
            }

        finally:
            # Clean up temporary file
            import os
            if os.path.exists(temp_pdf_path):
                os.unlink(temp_pdf_path)

    except Exception as e:
        print(f"❌ Error in get_book_chunks tool: {str(e)}")
        return {
            "error": str(e),
            "success": False
        }


# Update agent initialization
model = OpenAIModel(
    client_args={
        "api_key": os.environ["OPENAI_API_KEY"]
    },
    model_id="gpt-4o",
    params={
        "temperature": 0,
    }
)

agent = Agent(
    model=model,
    system_prompt=(
        "You are a book analyzer who will help to understand the structure and format of a book. "
        "Ignore preface, foreword, introduction, appendices, index, bibliography, and non-text content."
    ),
    tools=[get_book_chunks]
)


async def stream_book_analysis(book_id: str) -> AsyncGenerator[str, None]:
    """
    Streams the book analysis process in real-time with chunked processing.

    Args:
        book_id: The ID of the book to analyze

    Yields:
        Server-Sent Events (SSE) formatted strings with analysis progress
    """
    # Multi-step user instruction for chunked analysis with rate limiting
    user_instruction = f"""
    Analyze the book with ID: {book_id}

    STEP 1: Call the get_book_chunks tool with book_id="{book_id}" to get the book content in chunks.

    STEP 2: Process chunks in BATCHES to respect rate limits:
    - Process maximum 5 chunks at a time
    - Wait 60 seconds between batches
    - For each chunk, count "Chapter" and "Part" markers
    - Keep running totals

    STEP 3: After analyzing ALL chunks, provide the final count.

    STEP 4: Return ONLY this JSON format:
    {{
        "file_name": "<file_name from tool>",
        "file_id": "{book_id}",
        "pages": <page_count from tool>,
        "chapters": <total chapters across all chunks>,
        "parts": <total parts across all chunks, or null if none>
    }}

    IMPORTANT: 
    - Process chunks in small batches with delays to avoid rate limits
    - If you encounter rate limit errors, wait and retry
    - If the book has NO chapters OR NO parts, return: {{"error": "Book does not meet the format"}}

    Begin by calling get_book_chunks now.
    """

    try:
        # Send initial status
        yield f"data: {json.dumps({'status': 'started', 'message': 'Initializing book analysis...'})}\n\n"

        # Send downloading status
        yield f"data: {json.dumps({'status': 'downloading', 'message': 'Fetching and chunking book content...'})}\n\n"

        # Call agent
        result = agent(user_instruction)

        # Debug: Log the result object structure
        yield f"data: {json.dumps({'status': 'debug', 'message': f'Result type: {type(result).__name__}'})}\n\n"

        # Send analyzing status
        yield f"data: {json.dumps({'status': 'analyzing', 'message': 'Processing chunks and counting markers...'})}\n\n"

        # Extract text from AgentResult
        accumulated_text = ""

        # Try different approaches to extract result
        if hasattr(result, 'text') and result.text:
            accumulated_text = result.text
            yield f"data: {json.dumps({'status': 'debug', 'message': 'Found result.text'})}\n\n"

        elif hasattr(result, 'content') and result.content:
            content = result.content
            if isinstance(content, str):
                accumulated_text = content
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and 'text' in item:
                        accumulated_text += item['text']
                    elif hasattr(item, 'text'):
                        accumulated_text += item.text
            yield f"data: {json.dumps({'status': 'debug', 'message': 'Found result.content'})}\n\n"

        elif hasattr(result, 'messages') and result.messages:
            for msg in result.messages:
                if hasattr(msg, 'content'):
                    content = msg.content
                    if isinstance(content, list):
                        for item in content:
                            if isinstance(item, dict) and 'text' in item:
                                accumulated_text += item['text']
                            elif hasattr(item, 'text'):
                                accumulated_text += item.text
                    elif isinstance(content, str):
                        accumulated_text += content
            yield f"data: {json.dumps({'status': 'debug', 'message': 'Found result.messages'})}\n\n"

        elif hasattr(result, 'message') and result.message:
            message = result.message
            if hasattr(message, 'content'):
                content = message.content
                if isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict) and 'text' in item:
                            accumulated_text += item['text']
                        elif hasattr(item, 'text'):
                            accumulated_text += item.text
                        elif isinstance(item, str):
                            accumulated_text += item
                elif isinstance(content, str):
                    accumulated_text = content
            elif isinstance(message, str):
                accumulated_text = message
            yield f"data: {json.dumps({'status': 'debug', 'message': 'Found result.message'})}\n\n"

        else:
            accumulated_text = str(result)
            yield f"data: {json.dumps({'status': 'debug', 'message': 'Using str conversion'})}\n\n"

        # Send extracted content preview
        content_preview = accumulated_text[:1000] if accumulated_text else "NO CONTENT EXTRACTED"
        yield f"data: {json.dumps({'type': 'text', 'content': content_preview})}\n\n"

        # Process the final accumulated response
        yield f"data: {json.dumps({'status': 'processing', 'message': 'Parsing results...'})}\n\n"

        result_text = accumulated_text.strip()

        # Try to extract JSON from the response
        if "```json" in result_text:
            json_start = result_text.find("```json") + 7
            json_end = result_text.find("```", json_start)
            result_text = result_text[json_start:json_end].strip()
        elif "```" in result_text:
            json_start = result_text.find("```") + 3
            json_end = result_text.find("```", json_start)
            result_text = result_text[json_start:json_end].strip()
        elif "{" in result_text and "}" in result_text:
            json_start = result_text.find("{")
            json_end = result_text.rfind("}") + 1
            result_text = result_text[json_start:json_end].strip()

        try:
            # Parse and send final result
            parsed_result = json.loads(result_text)
            yield f"data: {json.dumps({'status': 'completed', 'result': parsed_result})}\n\n"

        except json.JSONDecodeError:
            # Send raw response if parsing fails
            yield f"data: {json.dumps({'status': 'completed', 'result': {'raw_response': accumulated_text}})}\n\n"

    except Exception as e:
        # Send error
        yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    finally:
        # Send done signal
        yield "data: [DONE]\n\n"


@app.get("/book/staging/{book_id}/analyze")
async def book_analysis_agent(book_id: str):
    """
    Streams the book analysis process in real-time using Server-Sent Events (SSE).

    Args:
        book_id: The ID of the book to analyze

    Returns:
        StreamingResponse with SSE formatted data
    """
    return StreamingResponse(
        stream_book_analysis(book_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )