from fastapi import FastAPI
from langchain.chains import LLMChain
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document
from pydantic import BaseModel
from tinydb import TinyDB, Query
from fastapi.middleware.cors import CORSMiddleware

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
    allow_methods=["GET","POST"],
    allow_headers=["*"],               # Allow all headers
)

class ChapterSummary(BaseModel):
    chapter_summary: str
    summary_option: str

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

    return chain.run(context=context_chapter_summary, question=question)

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
    crescent_city_db = None
    if book_name != "Select a Book" and chapter_name != "Select a Chapter":
        if book_name == "Crescent-City-Book-1":
            crescent_city_db = TinyDB('sm-crescent-city-book-1.json')
    return crescent_city_db.get(Chapter.Name == chapter_name)["Page Content"]

# API to fetch the chapter titles based on book selection
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
                    return [chapter["Name"] for chapter in chapter_docs]
    return {"error": "No chapters found for the selected book."}

# API to generate chapter summary based on selected option
@app.post("/chapter/summary")
def generate_chapter_summary(summary_with_option: ChapterSummary):
    """
    Generate a summary for the given chapter context based on the selected summary option.
    """
    if summary_with_option.summary_option == "Summary 1 - Summarize entire chapter using regular ChatGPT":
        return summarize_with_gpt4turbo(summary_with_option.chapter_summary, summary_with_option.summary_option)
    elif summary_with_option.summary_option == "Summary 2 - Summarize chapter part by part and merge":
        return summarize_with_langchain(summary_with_option.chapter_summary)
    elif summary_with_option.summary_option == "Summary 3 - Merge Summary 1 and Summary 2 using regular ChatGPT":
        return summarize_with_gpt4turbo(summary_with_option.chapter_summary, summary_with_option.summary_option)
    else:
        return {"error": "Invalid summary option selected."}

