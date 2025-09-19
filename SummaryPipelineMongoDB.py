import os
import os.path
import re
import time

import tiktoken
from PyPDF2 import PdfReader, PdfWriter
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

embedding_model = "nomic-embed-text:v1.5"

def summarize_with_langchain(text: str) -> str:
    # Use LangChain's summarization chain
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.1)
    docs = [Document(page_content=text)]
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    result = chain.run(docs)
    time.sleep(61)
    return result.strip()


def count_tokens(text, model_name="gpt-4-turbo"):
    # Get encoding for the model
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    return len(tokens)

def extract_chapter_docs():
    if not os.path.exists(extracted_pdf_path):
        extract_relevant_pages()
    # Load the PDF document
    loader = PyPDFLoader(extracted_pdf_path)
    documents = loader.load()

    import re
    chapter_docs = []
    section_pattern = r"PART\s+([A-Z]+\nTHE\s[A-Z]+)"
    chapter_pattern = r"[0-9]\s[0-9]*"
    section_title = ""
    chapter_title = ""
    for doc in documents:
        match = re.search(section_pattern, doc.page_content[0:40])
        if match:
            section_title_metadata_cleaned = match.group().replace("\n"," ")
            # section_title_metadata_cleaned = " ".join(section_title_metadata.split()[:-1])
            section_title = section_title_metadata_cleaned
        else:
            section_title_metadata_cleaned = section_title

        chapter_num_text = doc.page_content[:3]
        match = re.search(chapter_pattern, chapter_num_text)
        if match:
            chapter_title_metadata = "Chapter "+match.group().replace(" ","")
            chapter_title = chapter_title_metadata
        else:
            chapter_title_metadata = chapter_title

        doc.metadata = {
            "Part:":section_title_metadata_cleaned,
            "Chapter:":chapter_title_metadata,
            "Chapter Body/Passage:": doc.page_content
        }
        chapter_docs.append(doc)

    return chapter_docs

def extract_relevant_pages():
    if not os.path.exists(extracted_pdf_path):
        reader = PdfReader(pdf_path)
        writer = PdfWriter()
        for page in range(23, 749):
            writer.add_page(reader.pages[page])
        with open(extracted_pdf_path, "wb") as output_pdf:
            writer.write(output_pdf)
        print(f"Extracted pages 23 to 749 from House_of_Earth_and_Blood and saved to {extracted_pdf_path}")
    else:
        print("Extracted pages 23 to 749. Skipping to next step")

TOKEN_LIMIT = 128000
CHAR_LIMIT = TOKEN_LIMIT * 3

def extract_chapter_summary():
    chapter_documents = extract_chapter_docs()
    # Create a Dict of Chapter-Name: Summary of all chapter pages - Use OpenAI API to get summary
    # Loop through original docs and for entry matching Chapter-Name , retrieve the summary from Dict
    # and update metadata as 'summary'
    chapter_page_content = {}
    all_page_contents = None
    old_chapter_name = None
    key = None
    
    chapter_documents_with_chapter = list(filter(lambda doc: doc.metadata["Chapter:"] != '',  chapter_documents))
    for doc in chapter_documents_with_chapter:
        chapter_name = doc.metadata["Chapter:"]
        section_name = doc.metadata["Part:"]

        if old_chapter_name is None:
            old_chapter_name = chapter_name
            key = section_name + "-" + old_chapter_name

        # Update the chapter_page_content dict when new Chapter begins
        if old_chapter_name != chapter_name and key:
            chapter_page_content[key] = all_page_contents
            old_chapter_name = chapter_name
            key = section_name + "-" + old_chapter_name
            # Start with new Chapter page content
            # This is needed to exclude the chapter number during summarization
            start_index = get_start_index(doc.page_content)
            all_page_contents = doc.page_content[start_index:]
        else:
            # This is needed to exclude the chapter number during summarization
            start_index = get_start_index(doc.page_content)

            if all_page_contents is None:
                all_page_contents = doc.page_content[start_index:]
            else:
                all_page_contents += "\n" + doc.page_content[start_index:]


    # Populate the page contents of last chapter
    chapter_page_content[key] = all_page_contents

    filtered_dict = {
        k: v for k, v in chapter_page_content.items() if "Chapter" in k
    }
    from pymongo import MongoClient

    import urllib.parse
    username = urllib.parse.quote_plus("sm-user")
    password = urllib.parse.quote_plus("Platinum!56")
    uri = f"mongodb+srv://{username}:{password}@dev-cluster.dqw7iff.mongodb.net/?retryWrites=true&w=majority&appName=dev-cluster"

    client = MongoClient(uri)
    db = client["sarah-maas-db"]
    collection_books = db["sarah-maas-books"]

    for sec_chap_name, page_content in filtered_dict.items():
        name = sec_chap_name[sec_chap_name.index("-") + 1:]
        part = sec_chap_name[:sec_chap_name.index("-")]
        chapter_content = collection_books.find_one({"Name": name})
        if chapter_content:
            print(f"Chapter {sec_chap_name} exists in MongoDB")
        else:
            collection_books.insert_one({
                "Name": name,
                "Part": part,
                "Page Content": page_content
            })

 
def store_as_vectors(enriched_docs, vs_path: str):
    
    if not os.path.exists(vs_path):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_chapter_docs = text_splitter.split_documents(enriched_docs)
        
        # Create embeddings for the chunks
        embeddings = OllamaEmbeddings(model=embedding_model)
        print("Created Embeddings")
        # Create a vector store from the chunks
        vector_store = FAISS.from_documents(
            split_chapter_docs,
            embeddings
        )
        print("Created Vector Store")
        # Save the vector store to disk
        vector_store.save_local(vs_path)
        print(f"Saved vector store at {vs_path}")
    else:
        print(f"Path {vs_path} already exists.")


def get_start_index(page_content):
    digit_start = re.match(r'^\s*\d{1,2}\n*', page_content)
    if digit_start:
        return len(digit_start.group())
    else:
        return 0


if __name__ == "__main__":
    pdf_path = "/home/koushick/Young-Adult-ChatBot/Crescent-City/Maas_Sarah_J_-_Crescent_City_1_-_House_of_Earth_and_Blood.pdf"
    extracted_pdf_path = "/home/koushick/Young-Adult-ChatBot/Crescent-City/1-sm-cc-earth-and-blood-extracted.pdf"
    extract_chapter_summary()