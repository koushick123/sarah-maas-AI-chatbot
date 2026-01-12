# API to upload PDF
import asyncio
import json
import os
import urllib.parse
import uuid
from datetime import datetime
from fastapi import FastAPI
from fastapi import Form, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from gridfs import GridFS
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from pydantic import BaseModel
from pymongo import MongoClient

# Initialize FastAPI
app = FastAPI()
from DecryptCredentials import decrypt_openapi_key

os.environ["OPENAI_API_KEY"] = decrypt_openapi_key()  # Decrypt the OpenAI API key

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

UI_ORIGIN_URL = os.getenv("UI_ORIGIN_URL")

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

from DecryptCredentials import decrypt_mongo_user, decrypt_mongo_password, decrypt_mongo_hosturl

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
collection_book_chunk_metadata = db["sarah-maas-books-metadata"]

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


from bson import ObjectId
from BookAnalyzerEvents import book_analyzer_events

@app.get("/book/staging/{book_id}/analyze")
async def book_analysis(book_id: str):
    """
    Streams the book analysis process in real-time using Server-Sent Events (SSE).

    Args:
        book_id: The ID of the book to analyze

    Returns:
        StreamingResponse with SSE formatted data
    """

    return StreamingResponse(
        book_analyzer_events(book_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )


@app.delete("/book/staging/{book_id}/chunks/delete")
def delete_chunks(book_id: str):
    """
    Delete all chunks associated with a specific book_id from the database.

    Args:
        book_id: The unique identifier of the book whose chunks are to be deleted.
    """
    try:
        result = collection_book_chunks.delete_many({"book_id": book_id})
        print(f"{result}")
        print(f"Deleted {result.deleted_count} chunks for book_id: {book_id}")
        return {"success": True, "deleted_count": result.deleted_count, "book_id": book_id}
    except Exception as del_excep:
        print(f"Exception in delete = {str(del_excep)}")
        return {"success": False, "error": str(del_excep), "deleted_count": 0, "book_id": book_id}


@app.delete("/book/staging/{book_id}/chunk-metadata/delete")
def delete_chunk_metadata(book_id: str):
    """
    Delete chunk metadata associated with a specific book_id from the database.

    Args:
        book_id: The unique identifier of the book whose chunk metadata is to be deleted.
    """
    result = collection_book_chunk_metadata.delete_many({"file_id": book_id})
    print(f"Deleted {result.deleted_count} chunk metadata records for book_id: {book_id}")


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
    
    async def event_generator():
        try:
            # Send initial status
            yield f"data: {json.dumps({'status': 'started', 'message': 'Starting upload process'})}\n\n"
            await asyncio.sleep(1)

            # File already read
            yield f"data: {json.dumps({'status': 'reading', 'message': 'File content loaded', 'file_size': len(file_content)})}\n\n"
            await asyncio.sleep(1)

            # Generate unique book_id
            yield f"data: {json.dumps({'status': 'processing', 'message': 'Generating book ID'})}\n\n"
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            unique_id = str(uuid.uuid4())[:8]
            book_id = f"{book_name.lower().replace(' ', '-')}-{timestamp}-{unique_id}"

            print("Generated Book ID:", book_id)
            await asyncio.sleep(1)

            # Store PDF in GridFS
            yield f"data: {json.dumps({'status': 'uploading', 'message': 'Storing PDF in GridFS', 'book_id': book_id})}\n\n"
            file_id = fs.put(
                file_content,
                filename=f"{book_id}.pdf",
                book_id=book_id,
                content_type="application/pdf"
            )
            print("Stored file in GridFS with ID:", file_id)
            await asyncio.sleep(1)

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
            await asyncio.sleep(1)

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

#API to delete from staging
@app.delete("/book/staging/{book_id}/delete")
def delete_book_staging(book_id: str):
    """
    Delete staging record and associated PDF from GridFS.
    """
    try:
        # Find staging document
        doc = collection_book_staging.find_one({"book_id": book_id})
        if not doc:
            return {"error": "Book not found"}

        # Delete PDF from GridFS
        file_id = ObjectId(doc["file_id"])
        fs.delete(file_id)

        # Delete staging document
        collection_book_staging.delete_one({"book_id": book_id})

        return {"message": "Book staging record and PDF deleted successfully."}
    except Exception as e:
        return {"error": str(e)}


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

# FOR TESTING ONLY: Simple SSE endpoint that streams events every 2 seconds
from fastapi.responses import StreamingResponse
import time

async def generate_events():
    """Generator function that yields SSE formatted events"""
    count = 0
    while True:
        # Create event data
        data = {
            'message': f'Event number {count}',
            'timestamp': time.time(),
            'count': count
        }

        # Format as SSE event
        yield f"data: {json.dumps(data)}\n\n"

        count += 1
        await asyncio.sleep(2)


@app.get('/events')
async def events():
    """SSE endpoint that streams events to clients"""
    return StreamingResponse(
        generate_events(),
        media_type='text/event-stream',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',
            'Connection': 'keep-alive'
        }
    )


@app.get("/book/read_text_from_cropped_ocr_image/{image_path}")
def read_text_from_cropped_ocr_image(image_path: str):
    """
    Endpoint to read text from cropped OCR image.
    """
    from SarahMaasAzureOCR import read_text_from_cropped_ocr_image

    extracted_text = read_text_from_cropped_ocr_image(image_path)
    return extracted_text


@app.get("/book/decrypt_mongo_credentials")
def decrypt_mongo_credentials():
    """
    Endpoint to decrypt and return Mongo DB credentials.
    """
    from DecryptCredentials import decrypt_mongo_user, decrypt_mongo_password, decrypt_mongo_hosturl

    return JSONResponse(content={
    "mongo_user": decrypt_mongo_user(),
    "mongo_password": decrypt_mongo_password(),
    "mongo_hosturl": decrypt_mongo_hosturl()
    })