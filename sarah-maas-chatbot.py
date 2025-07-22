import base64
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import streamlit as st
from PyPDF2 import PdfReader, PdfWriter
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS

embedding_model = "nomic-embed-text:v1.5"

def store_pdf_as_vector_store(pdf_path, vs_path):

    if os.path.exists(vs_path):
        print(f"Vector store already exists at {vs_path}.")
    else:
        # Load the PDF document
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()

        # Split the document into smaller chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
        split_docs = text_splitter.split_documents(documents)

        # Create embeddings for the chunks
        embeddings = OllamaEmbeddings(model=embedding_model)

        # Create a vector store from the chunks
        vector_store = FAISS.from_documents(
            split_docs,
            embeddings
        )

        # Save the vector store to disk
        vector_store.save_local(vs_path)
        print(f"Vector store created and saved at {vs_path}.")
        return vector_store

# 🔄 Streamlit Callback Handler
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.tokens = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens += token
        self.container.markdown(self.tokens + "▌")  # blinking cursor effect

MAX_CONTEXT_TOKENS = 2048  # or less
MODEL_CONTEXT_LIMITS = {
    "llama3:8b": 8192,
    "llama3:8b-instruct-q4_0": 8192,
    "llama3:8b-q4_K_M": 8192,
    "mistral:7b": 8192,
    "mistral:7b-q4_K_M": 8192
}

def query_book(query, vector_store, model_name="llama3:8b-instruct-q4_0"):
    # Perform a similarity search
    results = vector_store.similarity_search(query, k=1)
    context = "\n".join([doc.page_content for doc in results])
    if len(context) > MODEL_CONTEXT_LIMITS[model_name]:
        context = context[:MAX_CONTEXT_TOKENS]

    # Create a prompt template for the LLM
    system_message = (
        "You are a knowledgeable literary research assistant with deep familiarity "
        "with Sarah J. Maas's *Throne of Glass* series. You help users analyze characters, plot, themes, symbols, and stylistic elements "
        "based on the provided text. Use the context from the books to answer the question with clarity and depth. "
        "You help in performing deconstructive analysis of the text, focusing on the internal contradictions. "
        "You analyze the text with literary senses like feminist, marxist, ecocritical perspectives, etc. "
        "If the context does not provide enough information, respond with: 'Sorry honey, I am not aware of this.'"
    )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(system_message),
        HumanMessagePromptTemplate.from_template(
            "Answer the following question using ONLY the context provided.\n"
            "Focus on offering thoughtful, research-level insight into the text.\n"
            "Avoid generic filler and do not add introductory phrases.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        ),
    ])

    # Set temperature based on model
    temperature = 0.8 if model_name == "HammerAI/mythomax-l2:latest" else 0.1

    # ⏳ Create a container to stream content
    stream_placeholder = st.empty()
    stream_handler = StreamlitCallbackHandler(stream_placeholder)

    # Correct instantiation
    selected_llm = Ollama(
        model=model_name,
        temperature=temperature,
        callbacks=[stream_handler]
    )

    chain = LLMChain(llm=selected_llm, prompt=prompt)
    llm_response = chain.run(context=context, question=query)
    return llm_response

def extract_pages_and_load_to_vector_store(input_pdf, book_spage_epage, book_pdf_name, vs_path):
    
    if os.path.exists(vs_path):
        print(f"Vector store for {book_pdf_name} already exists. Skipping extraction.")
        return None
    
    book_spage_epage = {
        k: v for k, v in book_spage_epage.items() 
        if k == book_pdf_name
    }
    book_sub_folder = list(book_spage_epage.keys())[0].removesuffix('.pdf')
    book_db_emdeddings_path = input_pdf + book_sub_folder
    vs_path = os.path.join(book_db_emdeddings_path, f"{book_pdf_name.removesuffix('.pdf')}_vector_store.faiss")
    pages = book_spage_epage[book_pdf_name]
    pdf_path = os.path.join(input_pdf, book_pdf_name)
    print(f"Processing {book_pdf_name} from {pdf_path}")
    start_page, end_page = pages
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    for page in range(start_page - 1, end_page):
        writer.add_page(reader.pages[page])
    output_pdf_path = os.path.join(book_db_emdeddings_path, f"{book_pdf_name.removesuffix('.pdf')}_extracted.pdf")
    with open(output_pdf_path, "wb") as output_pdf:
        writer.write(output_pdf)
    print(f"Extracted pages {start_page} to {end_page} from {book_pdf_name} and saved to {output_pdf_path}")
    print(f"Storing the extracted pages to vector store...{vs_path}")
    vector_store = store_pdf_as_vector_store(output_pdf_path, vs_path)
    return vs_path, vector_store

def load_vector_store(vs_path):
    if os.path.exists(vs_path):
        print(f"Loading vector store from {vs_path}")
        return FAISS.load_local(vs_path, OllamaEmbeddings(model=embedding_model), allow_dangerous_deserialization=True)

tog_book_title_vs_path_map = {}
tog_book_title_vector_store_data_map = {}

def process_pdf(idx, book_pdf_name):
    book_name_without_suffix = book_pdf_name[:-4]
    book_vector_path = os.path.join(root_pdf_path, book_name_without_suffix)
    vector_store_path = os.path.join(book_vector_path, f"{book_name_without_suffix}_vector_store.faiss")
    book_title = tog_title_list[idx]

    if os.path.exists(vector_store_path):
        vs = load_vector_store(vector_store_path)
        return book_title, vector_store_path, vs
    else:
        print(f"Vector store for {book_title} does not exist. Creating new vector store.")
        vs_path_vector_store = extract_pages_and_load_to_vector_store(
            root_pdf_path + "/", tog_book_spage_epage_map, book_pdf_name, vector_store_path
        )
        print(f"Vector store for {book_title} created at {vector_store_path}.")
        return book_title, vs_path_vector_store[0], vs_path_vector_store[1]

base64_encoded_images = {}

def get_base64_image(image_path):
    if image_path in base64_encoded_images:
        print(f"Image for {image_path} already encoded. Returning cached value.")
        return base64_encoded_images[image_path]
    with open(image_path, "rb") as img_file:
        print(f"Base64 encoding image {image_path}.")
        base64_image = base64.b64encode(img_file.read()).decode()
        base64_encoded_images[image_path]= base64_image
        return base64_image

if __name__ == "__main__":
    
    pdf_path_root = "/home/koushick/Young-Adult-ChatBot/"  # Replace with your PDF file path
    pdf_series = "Throne-Of-Glass"  # Replace with your PDF series name
    root_pdf_path = os.path.join(pdf_path_root, pdf_series)
    tog_book_spage_epage_map = {"1-sarah-maas-assassins-blade.pdf": [7, 318],
                                "2-sarah-maas-throne_of_glass.pdf": [7, 296],
                                "3-sarah-maas-Crown-of-Midnight.pdf": [8, 314],
                                "4-sarah_maas_heir_of_fire.pdf": [9, 426],
                                "5-sarah_maas_queen_of_shadows.pdf": [10, 545],
                                "6-sarah_maas-empire_of_storms.pdf": [15, 551],
                                "7-sarah_maas_tower_of_dawn.pdf": [10, 671],
                                "8-sarah-maas-kingdom-of-ash.pdf": [15, 2164]}

    tog_pdf_list = list(tog_book_spage_epage_map.keys())
    tog_title_list = [
        "Sarah Maas - Assassins Blade",
        "Sarah Maas - Throne of Glass",
        "Sarah Maas - Crown of Midnight",
        "Sarah Maas - Heir of Fire",
        "Sarah Maas - Queen of Shadows",
        "Sarah Maas - Empire of Storms",
        "Sarah Maas - Tower of Dawn",
        "Sarah Maas - Kingdom of Ash"
    ]
    # Dictionary to map book titles to their vector store paths   
    tog_title_cover_image_path_map = {}
    for index, title in enumerate(tog_title_list):
        tog_pdf_name = tog_pdf_list[index]
        tog_title_cover_image_path_map[title] = [tog_pdf_name, root_pdf_path + "/" + tog_pdf_name[0:len(tog_pdf_name)-4] + "/cover_page.png"]

    st.title("Sarah Maas AI Chatbot")
    st.write("Select your book and ask questions about it.")

    # Dropdown to select the book
    book_titles = ["Select a book"]
    # Populate the dropdown with book titles from the vector store
    for item in list(tog_title_cover_image_path_map.keys()):
        book_titles.append(item)

    with st.spinner("Loading the Books..."):
        # Use ThreadPoolExecutor to process PDFs concurrently
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for index, pdf_name in enumerate(tog_pdf_list):
                futures.append(executor.submit(process_pdf, index, pdf_name))

            for future in as_completed(futures):
                try:
                    title, path, vectorstore = future.result()
                    tog_book_title_vs_path_map[title] = path
                    tog_book_title_vector_store_data_map[title] = vectorstore
                except Exception as e:
                    print(f"Error processing one of the PDFs: {e}")

    st.success("Books Loaded. Ready to chat!")
    title_selection = st.selectbox("Select a book:", book_titles)
    user_input = ""

    if title_selection != "Select a book":
        img_base64 = get_base64_image(tog_title_cover_image_path_map[title_selection][1])
        st.markdown(
            f"""
            <div style='text-align: center;'>
                <img src="data:image/png;base64,{img_base64}" width="300"><br>
                <span><b>{title_selection}</b></span>
            </div>
            """,
            unsafe_allow_html=True
        )
        user_input = st.text_input("What would you like to know about the book?")

    if user_input.strip() and title_selection != "Select a book":
        with st.spinner("Searching the book..."):
            # Query the vector store with the user's input
            response = query_book(user_input, tog_book_title_vector_store_data_map[title_selection])
