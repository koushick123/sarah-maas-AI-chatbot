import os
import tempfile
import urllib.parse
import urllib.parse
from typing import AsyncGenerator

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
SSL_FLAG = os.getenv("SSL_FLAG", "false")
# Update VAUTL_ADDR as per Prod if SSL_FLAG is true
VAULT_ADDR = "<prod-url>"
VAULT_RETRIEVER_ADDR = os.getenv("VAULT_RETRIEVER_ADDR")
book_chunks: list[str] = []

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
        url = f"http://localhost:8300/v1/sm-secrets/data/openapi_mongodb_credentials"
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
collection_book_chunks = db["sarah-maas-books-chunk"]
collection_book_chunk_metadata = db["sarah-maas-books-chunk-metadata"]

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


def chunk_text(text: str, chunk_size: int = 25000, overlap: int = 500, book_id: str = "") -> int:    
    """
    Split text into chunks of specified size with overlap to preserve context and store in MongoDB.

    Args:
        text: The full text to chunk
        chunk_size: Maximum characters per chunk (default 25000 to leave room for instructions)
        overlap: Number of characters to overlap between chunks for context
        book_id: The unique identifier of the book

    Returns:
        Count of text chunks
    """
    if len(text) <= chunk_size:
        # Store single chunk in MongoDB if not exists
        if not collection_book_chunks.find_one({"index": 0}):
            collection_book_chunks.insert_one({
                "index": 0,
                "book_id": book_id,
                "chunk": text
            })
        return 1

    start = 0
    chunk_index = 0

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

        chunk_text_value = text[start:end]
        print("Appending chunk:", start, "to", end," with size ", len(chunk_text_value))
        
        # Store chunk in MongoDB if not exists
        if not collection_book_chunks.find_one({"index": chunk_index}):
            print("Inserting chunk index:", chunk_index)
            collection_book_chunks.insert_one({
                "index": chunk_index,
                "book_id": book_id,
                "chunk": chunk_text_value
            })
        
        chunk_index += 1

        if end < len(text):
            start = end - overlap
        else:
            start = end

    return chunk_index - 1

@tool
def get_chunk(index: int) -> str:
    book_chunk = collection_book_chunks.find_one({"index": index})
    if not book_chunk:
        return f"No Chunk found for index {index}"

    return book_chunk["chunk"]


@app.get("/book/staging/{book_id}/chunks")
@tool
def get_book_chunks(book_id: str, chunk_size: int = 25000):
    """
    Download PDF and return chunked text for analysis.
    This tool fetches a book PDF, extracts text, and chunks it for processing.

    Args:
        book_id: The unique identifier of the book to download
        chunk_size: Maximum characters per chunk

    Returns:
        Dictionary with chunks, metadata, and chunk info
    """

    try:
        print(f"🔧 TOOL CALLED: get_book_chunks with book_id={book_id}")
        print(f"Check if book chunks already exist for this book_id : {book_id}")

        book_doc = collection_book_chunk_metadata.find_one({"file_id": book_id})
        if book_doc:
            # Convert ObjectId fields to strings
            for key, value in book_doc.items():
                if isinstance(value, ObjectId):
                    book_doc[key] = str(value)
            return book_doc
        else:
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
                from PyPDF2 import PdfReader
            except Exception as e:
                raise RuntimeError("PyPDF2 is required to extract PDF text: " + str(e))

            try:
                # Open PDF using file path with PyPDF2
                with open(temp_pdf_path, "rb") as pdf_document:
                    reader = PdfReader(pdf_document)
                    page_count = len(reader.pages)
                    print("Extract text from PDF...")
                    for page_num in range(int(page_count)):
                        if page_num > 30:
                            return {"error": "Page number exceeded 30. Please revise the prompt to find page number for first chapter.", "success": False}
                        page = reader.pages[page_num]
                        print(f"Page number: {page_num + 1}")
                        page_text = page.extract_text() or ""
                        print(f"Page Text ====== {page_text[:100]}")
                        page_is_first_chap_obj = json.loads(identify_page_for_first_chapter(page_text))
                        print(f"First chapter page check result: {page_is_first_chap_obj}")
                        if page_is_first_chap_obj["first_chapter_page"]:
                            first_page_num = page_num + 1
                            break

                print(f"First chapter starts on page: {first_page_num}")
                print(f"📖 Extracting text from {first_page_num} page till {int(page_count)}...")

                with open(temp_pdf_path, "rb") as pdf_document:
                    full_text = ""
                    reader_for_full_text = PdfReader(pdf_document)
                    for page_num in range(first_page_num, int(page_count)):
                        page = reader_for_full_text.pages[page_num]
                        full_text += page.extract_text() or ""

                # Chunk the text
                print(f"Size of full text to chunk: {len(full_text)} characters")
                chunk_index = chunk_text(full_text, chunk_size, 200, book_id)
                print(f"📦 Text divided into {(chunk_index+1)} chunks")

                book_metadata = {
                    "file_id": book_id,
                    "file_name": f"{book_id}.pdf",
                    "page_count": page_count,
                    "page_for_first_chapter": first_page_num,
                    "total_chunks": (chunk_index + 1),
                    "total_characters": len(full_text)
                }

                collection_book_chunk_metadata.insert_one(book_metadata)
                for key, value in book_metadata.items():
                    if isinstance(value, ObjectId):
                        book_metadata[key] = str(value)
                return book_metadata
            finally:
                # Clean up temporary file
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
    model_id="gpt-5-nano",
    params={
        "temperature": 1,
    }
)

agent = Agent(
    model=model,
    system_prompt=(
        "You are a book analyzer who will help to understand the structure and format of a book. "
        "Ignore preface, foreword, introduction, appendices, index, bibliography, and non-text content."
    ),
    tools=[get_book_chunks, get_chunk]
)

import re


def identify_page_for_first_chapter(page_value: str):
    """
    Identify the page number where the first chapter starts.

    Args:
        page_value: Page contents to analyze

    Returns:
        Page number of the first chapter or -1 if not found
    """
    system_instruction = """
    You are a book analyzer tasked with identifying the page number where the first chapter begins.
    You MUST respond with valid JSON only, no additional text.
    """

    user_instruction = """
     1. For {page_value} contents check for below conditions:
 CONDITIONS:
     CONDITION 1: THE FIRST CHARACTER SHOULD BE ANY ONE OR A COMBINATION OF OCCURRENCE OF BELOW:
     2.1 Chapter
     2.2 CHAPTER
     2.3 number
     2.4 roman number
     2.5 number in text (case insensitive)
     
     CONDITION 2: CONDITION 1 SHOULD OCCUR EXACTLY ONCE ONLY
     
    ONLY IF CONDITION 1 AND CONDITION 2 ARE SATISFIED, return true, else false.

    3. If return true, exit the analysis and do not check further pages.
    OUTPUT SHOULD BE ONLY BE IN BELOW JSON FORMAT (no markdown, no code blocks, no extra text):
    {{
        "first_chapter_page": <true or false>, 
        "error": <any error messages encountered during analysis or null>
    }}
    """

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_instruction),
        HumanMessagePromptTemplate.from_template(user_instruction),
    ])

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = LLMChain(llm=llm, prompt=prompt)

    try:
        result = chain.run(page_value=page_value).strip()
        return result
    except Exception as e:
        print(f"Error during chain.run(): {type(e).__name__}: {str(e)}")
        raise


async def stream_book_analysis(book_id: str, chunk_size: int) -> AsyncGenerator[str, None]:
    """
    Streams the book analysis process in real-time with chunked processing.

    Args:
        book_id: The ID of the book to analyze
        chunk_size: Total number of chunks to process

    Yields:
        Server-Sent Events (SSE) formatted strings with analysis progress
    """
    # Multi-step user instruction for chunked analysis with rate limiting
    user_instruction = f"""
    Analyze the book with ID: {book_id}
    STEP 0: Call the get_book_chunks tool with book_id="{book_id}" to get the book content in chunks ONLY ONCE. 
    STEP 1: Start an index = -1
    
    Accumulate chapters and parts across all analyzed chunks (i.e., sum counts from each ANALYZER run after chunks )

    START LOOP :
    STEP 2: Set chapter_count with initial value of 0, part_count with initial value of 0.
    STEP 3: Increment index by 1
    STEP 4: Retrieve a chunk from get_chunk() by passing index
    STEP 5: If (index >= 7 AND (index % 7 is 0 OR index >= {chunk_size})), go to ANALYZER, else continue to STEP 3.

    ANALYZER: 
    1. Find out the number of chapters returned from the get_chunk() into chapter_count. Each Chapter begins with either "Chapter " or "CHAPTER " 
    or a number that is underlined, or a number that is highlighted in color.
    2. Update the chapter_count.
    3. Find out the number of parts returned from the get_chunk() into part_count. Each Part begins with either "Part " or "PART " or "Section" or "SECTION".
    4. Update the part_count.

    STEP 6: Remove all the text retrieved from get_chunk() called from STEP 3 from user instruction 
    passed to LLM to ensure rate limit or model context window size is not exceeded.
    STEP 7: Wait for 1 minute.
    STEP 8: Repeat STEP 3 (START LOOP) to STEP 7 until index = {chunk_size}.
    END LOOP

    STEP 9: Return ONLY this JSON format:
    {{
        "file_name": "<file_name from tool>",
        "file_id": "{book_id}",
        "chapters": <chapter_count>,
        "parts": <part_count>,
        "pages": <page_count from tool>,
        "error": <any error messages encountered during analysis or null>
    }}
    Begin by calling get_book_chunks now.
    DO NOT PROVIDE INTERMEDIATE RESPONSES. WAIT UNTIL ALL STEPS ARE COMPLETED BEFORE RESPONDING.
    DO NOT RETURN ANYTHING OTHER THAN THE JSON IN STEP 10. ANY ERRORS MUST ONLY BE REPORTED IN THE "error" FIELD OF THE JSON.
    """

    try:
        # Send initial status
        yield f"data: {json.dumps({'status': 'started', 'message': 'Initializing book analysis...'})}\n\n"

        # Send downloading status
        yield f"data: {json.dumps({'status': 'downloading', 'message': 'Fetching and chunking book content...', 'total_chunks': chunk_size})}\n\n"

        # Send analyzing status
        yield f"data: {json.dumps({'status': 'analyzing', 'message': 'Processing chunks and analyzing structure...'})}\n\n"

        # Call agent and capture result
        result = agent(user_instruction)

        # Parse the result to extract the JSON from STEP 10
        analysis_result = None

        # Try to extract JSON from the result
        if isinstance(result, dict):
            analysis_result = result
            print("Analysis Result Dict:", analysis_result)
        elif isinstance(result, str):
            # Try to find JSON in the response
            json_match = re.search(r'\{[^{}]*"file_name"[^{}]*\}', result, re.DOTALL)
            if json_match:
                try:
                    analysis_result = json.loads(json_match.group())
                except json.JSONDecodeError:
                    # If parsing fails, try to extract manually
                    pass

        if analysis_result:
            # Send completion status with the analysis result
            yield f"data: {json.dumps({'status': 'completed', 'message': 'Book analysis completed successfully', 'result': analysis_result})}\n\n"
        else:
            # If we couldn't parse the result, send it as-is
            yield f"data: {json.dumps({'status': 'completed', 'message': 'Book analysis completed', 'raw_result': str(result)})}\n\n"

    except Exception as e:
        # Send error with details
        error_message = str(e)
        yield f"data: {json.dumps({'status': 'error', 'message': f'Analysis failed: {error_message}'})}\n\n"

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
        stream_book_analysis(book_id,56),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )

# API to upload PDF
from fastapi import Form, File, UploadFile
from fastapi.responses import StreamingResponse
from datetime import datetime
import uuid
import json
import asyncio


@app.post("/book/staging/upload")
async def upload_book_pdf(
        book_name: str = Form(...),
        file: UploadFile = File(...)
):
    """
    Upload PDF file to GridFS and create staging record with SSE progress.
    """
    # Read file content BEFORE the generator starts
    file_content = await file.read()
    original_filename = file.filename

    async def event_generator():
        try:
            # Send initial status
            yield f"data: {json.dumps({'status': 'started', 'message': 'Starting upload process'})}\n\n"
            await asyncio.sleep(0.1)

            # File already read
            yield f"data: {json.dumps({'status': 'reading', 'message': 'File content loaded', 'file_size': len(file_content)})}\n\n"
            await asyncio.sleep(0.1)

            # Generate unique book_id
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Generating book ID'})}\n\n"
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            book_id = f"{book_name.lower().replace(' ', '-')}-{timestamp}-{unique_id}"

            print("Generated Book ID:", book_id)
            await asyncio.sleep(0.1)

            # Store PDF in GridFS
            yield f"data: {json.dumps({'status': 'uploading', 'message': 'Storing PDF in GridFS', 'book_id': book_id})}\n\n"
            file_id = fs.put(
                file_content,
                filename=f"{book_id}.pdf",
                book_id=book_id,
                content_type="application/pdf"
            )
            print("Stored file in GridFS with ID:", file_id)
            await asyncio.sleep(0.1)

            # Create staging document
            yield f"data: {json.dumps({'status': 'saving', 'message': 'Creating staging record'})}\n\n"
            document = {
                "book_name": book_name,
                "book_id": book_id,
                "file_id": str(file_id),
                "file_size": len(file_content),
                "content_type": "application/pdf"
            }

            result = collection_book_staging.insert_one(document)
            await asyncio.sleep(0.1)

            # Send success response
            yield f"data: {json.dumps({'status': 'completed', 'message': 'PDF uploaded successfully', 'book_id': book_id, 'file_id': str(file_id), 'mongo_id': str(result.inserted_id), 'file_size': len(file_content)})}\n\n"

        except Exception as e:
            # Send error response
            yield f"data: {json.dumps({'status': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


# API to download PDF
@app.get("/book/staging/{book_id}/download")
def download_book_pdf(book_id: str):
    """
    Download PDF file from GridFS.
    """
    try:
        # Get staging document
        doc = collection_book_staging.find_one({"book_id": book_id})
        if not doc:
            return {"error": "Book not found"}

        # Retrieve PDF from GridFS
        file_id = ObjectId(doc["file_id"])
        pdf_file = fs.get(file_id)

        return Response(
            content=pdf_file.read(),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={book_id}.pdf"
            }
        )
    except Exception as e:
        return {"error": str(e)}