from fastapi import FastAPI
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from pydantic import BaseModel
from tinydb import TinyDB, Query
from fastapi.middleware.cors import CORSMiddleware
from cryptography.fernet import Fernet
import os

# Initialize FastAPI app
app = FastAPI()

# Allow specific origins (replace with your Angular dev server URL)
origins = [
    "http://localhost:4200"  # Angular local dev
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

def decryptopenapi():
    """
    Function to decrypt OpenAPI key.
    """
    encryptionkey = "f1kneQl7vqezzY8GXWDRLl1cXdImiyQYKVNOf4thQhM="
    if not encryptionkey:
        raise ValueError("FERNET_KEY environment variable not set")
    fernet = Fernet(encryptionkey)
    with open("encryptedopenapi.txt") as file:
        encrypted_api = file.read().encode()
    return fernet.decrypt(encrypted_api).decode()

os.environ["OPENAI_API_KEY"] = decryptopenapi()  # Decrypt the OpenAI API key

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
        "Avoid generic filler and do not add introductory phrases."
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
    if option == "Summary 1 - Summarize entire chapter using regular ChatGPT":
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

# API to fetch the chapter contents based on book and chapter selection
@app.get("/book/{book_name}/chapter/{chapter_name}/contents")
def fetch_book_contents(book_name : str, chapter_name: str):
    # Get the chapter contents as per selection
    Chapter = Query()
    if book_name != "Select a Book" and chapter_name != "Select a Chapter":
        if book_name == "Crescent-City-Book-1":
            crescent_city_db = TinyDB('sm-crescent-city-book-1.json')
            chapter_summary = crescent_city_db.get(Chapter.Name == chapter_name)["Page Content"]
            if int(chapter_name[chapter_name.index(" "):]) > 9:
                return chapter_summary[2:]
            else:
                return chapter_summary

# API to fetch the chapter and part titles based on book selection
@app.get("/book/{book_name}/chapters")
def fetch_chapter_titles(book_name: str):
    """
    Fetch chapter titles based on the selected book.
    """
    Chapter = Query()
    if book_name != "Select a Book":
        if book_name == "Crescent-City-Book-1":
            crescent_city_db = TinyDB('sm-crescent-city-book-1.json')
            if crescent_city_db:
                chapter_docs = crescent_city_db.search(Chapter.Name.exists())
                if chapter_docs:
                    part_chapter_map = {}
                    for chapter in chapter_docs:
                        part = chapter["Part"]
                        chapter_name = chapter["Name"]
                        if part not in part_chapter_map:
                            part_chapter_map[part] = []
                        part_chapter_map[part].append(chapter_name)
                    return part_chapter_map
                    # return [chapter["Name"] for chapter in chapter_docs]
    return {"error": "No chapters found for the selected book."}

# API to save the chapter summary
@app.post("/chapter/save")
def save_chapter_summary(chapter_summary: ChapterSummary):
    """
    Save the chapter summary to the database.
    """
    chapterDetail = Query()
    if chapter_summary.chapter_summary and chapter_summary.summary_option and chapter_summary.book_name\
            and chapter_summary.part and chapter_summary.chapter_name:
        crescent_city_db = TinyDB('sm-crescent-city-book-1-summary.json')
        crescent_city_db.upsert({
            "Name": chapter_summary.chapter_name,
            "Part": chapter_summary.part,
            "Summary Option": chapter_summary.summary_option,
            "Book Name": chapter_summary.book_name,
            "Summary": chapter_summary.chapter_summary
        },
        (chapterDetail.Name == chapter_summary.chapter_name & chapterDetail.Book_Name == chapter_summary.book_name & chapterDetail.Part == chapter_summary.part
                                                                & chapterDetail.Summary_Option == chapter_summary.summary_option)
        )
        return {"message": "Chapter summary saved successfully."}
    else:
        return {"error": "Chapter summary details is missing."}

# API to generate chapter summary based on selected option
@app.post("/chapter/summary")
def generate_chapter_summary(chapter_content: Chapter):
    """
    Generate a summary for the given chapter context based on the selected summary option.
    """
    if chapter_content.summary_option == "summary1":
        # Fetch existing summary from the database if available
        crescent_city_db = TinyDB('sm-crescent-city-book-1-summary.json')
        Chapter = Query()
        existing_summary = crescent_city_db.get(Chapter.Name == chapter_content.chapter_content and Chapter.book_name == chapter_content.book_name
                                                and Chapter.Part == chapter_content.part
                                                and Chapter.Summary_Option == chapter_content.summary_option)
        if existing_summary:
            print(f"Existing summary found for {chapter_content.chapter_name}")
            return {"summary": existing_summary["Summary"]}
        print("No existing summary found, generating new summary...")
        return summarize_with_gpt4turbo(chapter_content.chapter_content, chapter_content.summary_option)
    elif chapter_content.summary_option == "summary2":
        return summarize_with_langchain(chapter_content.chapter_content)
    elif chapter_content.summary_option == "summary3":
        return summarize_with_gpt4turbo(chapter_content.chapter_content, chapter_content.summary_option)
    else:
        return {"error": "Invalid summary option selected."}
