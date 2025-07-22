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
import re

embedding_model = "nomic-embed-text:v1.5"

def store_pdf_as_vector_store(pdf_path, vs_path, book_title, timeline_arc, deconstructive_analysis):

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

        if timeline_arc:
            # 🆕 Enrich chunks with character timeline metadata
            enriched_docs = enrich_with_metadata(split_docs, book_title)

            # Create a vector store from the chunks
            vector_store = FAISS.from_documents(
                enriched_docs,
                embeddings
            )
        elif deconstructive_analysis:
            deconstruct_docs = enrich_with_deconstructive_analysis(documents, book_title)
            # Create a vector store from the chunks
            vector_store = FAISS.from_documents(
                deconstruct_docs,
                embeddings
            )
        else:
            # Create a vector store from the chunks
            vector_store = FAISS.from_documents(
                split_docs,
                embeddings
            )

        # Save the vector store to disk
        vector_store.save_local(vs_path)
        print(f"Vector store created and saved at {vs_path}.")
        return vector_store

def identify_deconstructed_metadata(book_title):
    if book_title == "Sarah Maas - Assassins Blade":
        return {
            "book_title": "Assassins Blade",
            "chapter": "The Stranger in the Mist",
            "type": "deconstruction",
            "tags": ["identity", "contradiction", "binary", "ambiguity"],
            "excerpt": "“Just like how you proper men surrounded a defenseless girl in an alley?”",
                     "analysis_note": "This passage exposes a contradiction between the stranger's words and actions, subverting earlier meanings about her character. "
                                      "The phrase 'proper men' is initially ambiguous, but Yrene's reaction suggests that it refers to those who would harm or exploit women. "
                                      "However, the stranger herself wields two long daggers with dripping blood, blurring the binary opposition between victimizer and protector. "
                                      "This ambiguity challenges the reader's expectations about her character, "
                                      "as she simultaneously critiques the behavior of 'proper men' while also embodying a violent and potentially oppressive power dynamic." }

    elif book_title == "Sarah Maas - Throne of Glass":
        return { "book": "Throne of Glass",
                 "type": "deconstruction",
                 "tags": ["binary", "ambiguity", "identity"],
                 "chapter": "Not specified",
                 "excerpt": "\"That he's very attached to her. Possibly in love with her.\"",
                 "analysis_note": "The passage exposes a binary opposition between the Duke's perception of Lady Lillian as 'foolish' and Kaltain's understanding of Dorian's "
                                  "attachment to her as 'tragic'. This contradiction highlights the ambiguity surrounding Lady Lillian's character identity, "
                                  "which is further complicated by Kaltain's own motivations. The Duke's dismissal of Dorian's feelings for Lady Lillian as 'impossible' "
                                  "suggests a breakdown in traditional meaning structures, implying that societal norms and expectations may not apply to this particular situation. "
                                  "This internal contradiction challenges the reader's understanding of the characters and their relationships, adding complexity to the narrative."}

    elif book_title == "Sarah Maas - Crown of Midnight":
        return { "book": "Crown of Midnight",
                 "type": "deconstruction",
                 "tags": ["identity", "contradiction", "binary", "ambiguity"],
                 "chapter": "The Wagon",
                 "excerpt": "\"I found a riddle, and my friends have been debating its answer for weeks. We even have a bet going about it,\" she said as vaguely as she could.",
                 "analysis_note": "This passage exposes an internal contradiction in Celaena's character identity. On one hand, she presents herself as a carefree and impudent"
                                  " individual who is willing to make a bet with Yellowlegs. This aligns with her earlier persona as a charming and confident assassin. "
                                  "However, the ambiguity surrounding her 'friends' and their motivations suggests that Celaena may be struggling with her own identity and loyalties. "
                                  "The binary opposition between her public image and private doubts creates tension and subverts the traditional meaning structure of a single, "
                                  "fixed character identity." }

    elif book_title == "Sarah Maas - Heir of Fire":
        return { "book": "Heir of Fire",
                 "type": "deconstruction",
                 "tags": ["binary opposition", "ambiguity", "character identity"],
                 "chapter": "Chaol's Experiment with Dorian",
                 "excerpt": "I think that this kingdom could use a healer as its queen.",
                 "analysis_note": "The passage exposes a binary opposition between Chaol's perception of Sorscha and her actual character. On one hand, Chaol sees Sorscha as "
                                  "'truly stunning' and thinks she would make a great queen, implying a romantic interest. On the other hand, Sorscha is revealed to be sad "
                                  "and unresponsive to Chaol's suggestion, subverting his earlier expectations. This ambiguity challenges the traditional notion of a character's "
                                  "identity being fixed or easily readable, instead highlighting the complexity and nuance of human emotions." }

    elif book_title == "Sarah Maas - Queen of Shadows":
        return { "book": "Queen of Shadows",
                 "type": "deconstruction",
                 "tags": ["identity", "contradiction", "binary"],
                 "chapter": "The Blackbeaks",
                 "excerpt": "Good choice, witchling, Manon said, and the word was a challenge and an order. She turned away, but glanced over her shoulder. "
                            "Welcome to the Blackbeaks.",
                 "analysis_note": "This passage exposes a binary opposition between belonging and mistake, as Elide reflects on her decision to join the Blackbeaks. "
                                  "The phrase 'strange' suggests that this feeling of belonging is not entirely coherent with her earlier identity as a witchling. "
                                  "This internal contradiction highlights the complexity of Elide's character development, as she navigates the blurred lines "
                                  "between loyalty, duty, and personal desire. The ambiguity surrounding her decision to join the Blackbeaks subverts "
                                  "traditional notions of good vs. evil, instead presenting a nuanced exploration of moral gray areas." }

    elif book_title == "Sarah Maas - Empire of Storms":
        return { "book": "Empire of Storms",
                 "type": "deconstruction",
                 "tags": ["binary opposition", "ambiguity", "character identity"],
                 "chapter": "Ilium's Significance",
                 "excerpt": "The Mycenians are nothing more than a myth—they were banished three hundred years ago. If you’re looking for a symbol, they’re fairly "
                            "outdated—and divisive.",
                 "analysis_note": "This passage exposes a binary opposition between the perceived value of the Mycenians as a myth and their actual historical significance. "
                                  "The text initially presents the Mycenians as 'nothing more than a myth', implying that they are irrelevant and outdated. "
                                  "However, Aelin's subsequent explanation reveals that the Mycenians were once powerful crime lords who ruled Ilium and played a crucial role in "
                                  "winning wars. This binary opposition between myth and reality subverts earlier meanings by highlighting the complexity of historical narratives. "
                                  "Additionally, this passage also raises questions about character identity, as Aelin's connection to the Mycenians is ambiguous. "
                                  "Her cousin's calculation of reasons why Ilium is vital suggests that Aelin's motivations may be rooted in her own sense of identity and purpose." }

    elif book_title == "Sarah Maas - Tower of Dawn":
        return { "book": "Tower of Dawn",
                 "type": "deconstruction",
                 "tags": ["binary opposition", "ambiguity", "character identity"],
                 "chapter": "N/A",
                 "excerpt": "\"I don't blame them for abandoning it if it's this cold in the summer,\" Nesryn muttered. \"Imagine it in winter.\"",
                 "analysis_note": "The passage exposes a binary opposition between the harsh, unforgiving environment of the Fells and the perceived abandonment of the tower. "
                                  "This contrast subverts earlier meanings by suggesting that even in the most inhospitable places, there can be beauty (the 'cool' air) "
                                  "and value (the intact archway). Nesryn's muttering also hints at ambiguity in her own character identity, as she struggles to reconcile her "
                                  "initial relief with the potential danger of the tower. This internal contradiction highlights the complexity of Nesryn's personality and "
                                  "underscores the theme of uncertainty that pervades the novel."}

    elif book_title == "Sarah Maas - Kingdom of Ash":
        return { "book": "Kingdom of Ash",
                 "chapter": "26",
                 "type": "deconstruction",
                 "tags": ["identity", "contradiction", "binary"],
                 "excerpt": "When you finish breaking me apart for the day, how does it feel to know that you are still nothing?",
                 "analysis_note": "The passage exposes a contradiction in Cairn's character identity. On one hand, he grins and says 'Some fire left in you, it seems. "
                                  "Good.' suggesting a sense of admiration or even affection towards Aelin. However, this is immediately subverted by his subsequent statement "
                                  "that she is the only reason he has an oath to fulfill, implying that without her, he is nothing. This binary opposition between Cairn's "
                                  "perceived value and actual worth highlights the theme of identity crisis in the series. The ambiguity surrounding Cairn's character raises "
                                  "questions about the nature of loyalty, power dynamics, and the blurred lines between good and evil." }

def enrich_with_deconstructive_analysis(documents , book_title):
    deconstructed_docs = []
    for doc in documents:
        # Deconstructive analysis from the text
        metadata = identify_deconstructed_metadata(book_title)
        doc.metadata = metadata
        deconstructed_docs.append(doc)

    return deconstructed_docs


def enrich_with_metadata(documents, book_title):
    enriched_docs = []
    for doc in documents:
        page_text = doc.page_content
        # Identify character and timeline phase based on the text
        # Search for chapter title or number in the text
        chapter_title = ""
        if "Chapter" in page_text:
            chapter_title = page_text.split("Chapter")[1].split("\n")[0].strip()

        metadata = {
            "book_title": book_title,
            "chapter_title": chapter_title,
            "character": "Celaena" if "Celaena" in page_text or "Aelin" in page_text else "Other",
            "timeline_phase": identify_timeline_phase(book_title, page_text),
            "themes": identify_themes(page_text)
        }
        doc.metadata = metadata
        enriched_docs.append(doc)
    return enriched_docs


def extract_chapter_title(text):
    # Match common formats: "CHAPTER 4", "Chapter Four", "Chapter 5: The Woods"
    chapter_pattern = r"(?i)\bchapter\s+([0-9]+|[IVXLCDM]+|[A-Za-z]+)(?::\s*([\w\s',-]+))?"
    match = re.search(chapter_pattern, text)
    if match:
        chapter_num = match.group(1).strip()
        chapter_subtitle = match.group(2).strip() if match.group(2) else ""
        return f"Chapter {chapter_num}" + (f": {chapter_subtitle}" if chapter_subtitle else "")
    return ""


def identify_themes(text):
    themes = []
    if any(word in text for word in ["identity", "name", "past"]):
        themes.append("identity")
    if any(word in text for word in ["magic", "power", "control"]):
        themes.append("power")
    if any(word in text for word in ["death", "Sam", "trauma", "scars"]):
        themes.append("trauma")
    if any(word in text for word in ["Rowan", "trust", "loyalty", "court"]):
        themes.append("loyalty")
    return themes


def identify_timeline_phase(book_title, text):
    if "Sam" in text and "Guild" in text:
        return "assassin-training"
    elif book_title == "Sarah Maas - Heir of Fire" and "Rowan" in text:
        return "fae-training"
    elif "Queen" in text and "magic" in text:
        return "queen-awakening"
    elif "battle" in text and "Erawan" in text:
        return "final-war"
    else:
        return "unknown"


# 🔄 Streamlit Callback Handler
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.tokens = ""

    def on_llm_new_token(self, token: str, **kwargs):
        self.tokens += token
        render_scrollable_text(self.tokens + "▌")

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

    return query_with_context(query, context, model_name)


def query_all_books(query, vectorstore_map, model_name="llama3:8b-instruct-q4_0"):
    # 🔍 Search all vector stores and combine top-k results
    all_results = []
    for title, store in vectorstore_map.items():
        try:
            results = store.similarity_search(query, k=1)
            all_results.extend(results)
        except Exception as e:
            print(f"Error searching {title}: {e}")

    # 🧠 Combine and sort results (optional: sort by similarity score if supported)
    combined_context = "\n".join([doc.page_content for doc in all_results[:5]])

    # 🧩 Truncate if needed
    if len(combined_context) > MODEL_CONTEXT_LIMITS[model_name]:
        combined_context = combined_context[:MAX_CONTEXT_TOKENS]

    # Reuse existing prompt and LLM logic
    return query_with_context(query, combined_context, model_name)


def query_with_context(query, context, model_name):
    system_message = (
        "You are a knowledgeable literary research assistant with deep familiarity "
        "with Sarah J. Maas's *Throne of Glass* series. [...]"
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

    temperature = 0.1
    stream_placeholder = st.empty()
    stream_handler = StreamlitCallbackHandler(stream_placeholder)

    selected_llm = Ollama(
        model=model_name,
        temperature=temperature,
        callbacks=[stream_handler]
    )

    chain = LLMChain(llm=selected_llm, prompt=prompt)
    return chain.run(context=context, question=query)


def extract_pages_and_load_to_vector_store(input_pdf, book_spage_epage, book_pdf_name, vs_path, book_title, timeline_arc, deconstructive_analysis):
    
    if os.path.exists(vs_path):
        print(f"Vector store for {book_pdf_name} already exists. Skipping extraction.")
        return None
    
    book_spage_epage = {
        k: v for k, v in book_spage_epage.items() 
        if k == book_pdf_name
    }
    book_sub_folder = list(book_spage_epage.keys())[0].removesuffix('.pdf')
    book_db_emdeddings_path = input_pdf + book_sub_folder
    if deconstructive_analysis:
        vs_path = os.path.join(book_db_emdeddings_path, f"{book_pdf_name.removesuffix('.pdf')}_deconstructive_analysis_vector_store.faiss")
    elif timeline_arc:
        vs_path = os.path.join(book_db_emdeddings_path, f"{book_pdf_name.removesuffix('.pdf')}_timeline_arc_vector_store.faiss")
    else:
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
    vector_store = store_pdf_as_vector_store(output_pdf_path, vs_path, book_title, timeline_arc, deconstructive_analysis)
    return vs_path, vector_store


def load_vector_store(vs_path):
    if os.path.exists(vs_path):
        print(f"Loading vector store from {vs_path}")
        return FAISS.load_local(vs_path, OllamaEmbeddings(model=embedding_model), allow_dangerous_deserialization=True)


def process_pdf(idx, book_pdf_name, timeline_arc, deconstructive_analysis):
    if timeline_arc:
        book_name_without_suffix = book_pdf_name[:-4]
        book_vector_path = os.path.join(root_pdf_path_timeline, book_name_without_suffix)
        vector_store_path = os.path.join(book_vector_path,
                                         f"{book_name_without_suffix}_timeline_arc_vector_store.faiss")
    elif deconstructive_analysis:
        book_name_without_suffix = book_pdf_name[:-4]
        book_vector_path = os.path.join(root_pdf_path_deconstructive_analysis, book_name_without_suffix)
        vector_store_path = os.path.join(book_vector_path, f"{book_name_without_suffix}_deconstructive_analysis_vector_store.faiss")
    else:
        book_name_without_suffix = book_pdf_name[:-4]
        book_vector_path = os.path.join(root_pdf_path, book_name_without_suffix)
        vector_store_path = os.path.join(book_vector_path, f"{book_name_without_suffix}_vector_store.faiss")

    book_title = tog_title_list[idx]

    print(f"VS Path = {vector_store_path}")
    if os.path.exists(vector_store_path):
        vs = load_vector_store(vector_store_path)
        return book_title, vector_store_path, vs
    else:
        print(f"Vector store for {book_title} does not exist. Creating new vector store. Timeline Arc: {timeline_arc}. Deconstructive Analysis: {deconstructive_analysis}")
        if not timeline_arc:
            vs_path_vector_store = extract_pages_and_load_to_vector_store(root_pdf_path_timeline + "/",
                                                                          tog_book_spage_epage_map,
                                                                          book_pdf_name, vector_store_path, book_title,
                                                                          timeline_arc, deconstructive_analysis)
        elif deconstructive_analysis:
            vs_path_vector_store = extract_pages_and_load_to_vector_store(root_pdf_path_timeline + "/",
                                                                          tog_book_spage_epage_map,
                                                                          book_pdf_name, vector_store_path, book_title,
                                                                          timeline_arc, deconstructive_analysis)
        else:
            vs_path_vector_store = extract_pages_and_load_to_vector_store(root_pdf_path + "/", tog_book_spage_epage_map,
                                                                          book_pdf_name, vector_store_path, book_title,
                                                                          timeline_arc, deconstructive_analysis)
        #print(f"Vector store for {book_title} created at {vector_store_path}.")
        return book_title, vs_path_vector_store[0], vs_path_vector_store[1]


def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        print(f"Base64 encoding image {image_path}.")
        base64_image = base64.b64encode(img_file.read()).decode()
        base64_encoded_images[image_path]= base64_image
        return base64_image

def render_scrollable_text(text):
    output_container.markdown(f"<div class='scrollable-container'>{text}</div>", unsafe_allow_html=True)

def clear_user_input():
    st.session_state.user_input = ""  # Clear the user input when the search type changes

base64_encoded_images = {}
# tog_book_title_vs_path_map = {}
tog_book_title_vector_store_data_map = {}
tog_book_title_vector_store_data_map_timeline_arc = {}
pdf_path_root = "/home/koushick/Young-Adult-ChatBot/"  # Replace with your PDF file path
# PDF Path without Timeline Arc
pdf_series = "Throne-Of-Glass"  # Replace with your PDF series name
root_pdf_path = os.path.join(pdf_path_root, pdf_series)

#PDF Path with Timeline Arc
pdf_series_timeline_arc = "Throne-Of-Glass-Timeline-Arc"  # Replace with your PDF series name for timeline arc
root_pdf_path_timeline = os.path.join(pdf_path_root, pdf_series + "-timeline-arc")
root_pdf_path_deconstructive_analysis = os.path.join(pdf_path_root, pdf_series + "-deconstructive-analysis")

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

st.set_page_config(page_title="Sarah Maas AI Chatbot", layout="centered")
st.title("Sarah Maas AI Chatbot")
st.write("Welcome to the Sarah Maas AI Chatbot! I have been trained on the Throne of Glass series, including all the books and a timeline arc of Celaena's journey across the series. You can ask me questions about the books or the timeline arc.")
img_base64 = get_base64_image("Sarah-Maas-AI.png")
st.markdown(
    f"""
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{img_base64}" width="300"><br>
        </div>
        """,
    unsafe_allow_html=True
)

# Dropdown to select the book
book_titles = ["Select a book"]
# Populate the dropdown with book titles from the vector store
for item in list(tog_title_cover_image_path_map.keys()):
    book_titles.append(item)

with st.spinner("Loading the Books..."):
    # Use ThreadPoolExecutor to process PDFs concurrently
    # Process all PDFs normally
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for index, pdf_name in enumerate(tog_pdf_list):
            futures.append(executor.submit(process_pdf, index, pdf_name, False, False))

        for future in as_completed(futures):
            try:
                title, path, vectorstore = future.result()
                # tog_book_title_vs_path_map[title] = path
                tog_book_title_vector_store_data_map[title] = vectorstore
            except Exception as e:
                print(f"Error processing one of the PDFs: {e}")

    # Use ThreadPoolExecutor to process PDFs concurrently
    # Process all PDFs with timeline arc
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for index, pdf_name in enumerate(tog_pdf_list):
            futures.append(executor.submit(process_pdf, index, pdf_name, True, False))

        for future in as_completed(futures):
            try:
                title, path, vectorstore = future.result()
                # tog_book_title_vs_path_map_timeline_arc[title] = path
                tog_book_title_vector_store_data_map_timeline_arc[title] = vectorstore
            except Exception as e:
                print(f"Error processing one of the PDFs: {e}")

    # Process all PDFs for deconstructive analysis
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = []
        for index, pdf_name in enumerate(tog_pdf_list):
            futures.append(executor.submit(process_pdf, index, pdf_name, False, True))

        for future in as_completed(futures):
            try:
                title, path, vectorstore = future.result()
                # tog_book_title_vs_path_map_timeline_arc[title] = path
                tog_book_title_vector_store_data_map_timeline_arc[title] = vectorstore
            except Exception as e:
                print(f"Error processing one of the PDFs: {e}")

#st.success("Books Loaded. Ready to chat!")
st.write("Choose search type and explore the books.")
search_options = ["A Timeline Arc of Celeana across Throne of Glass Series", "Deconstructive Analysis of the Throne of Glass Series", "Specific book with Throne of Glass Series"]
search_type = st.radio("I would like to search ", [search_options[0], search_options[1], search_options[2]], index=0, on_change=clear_user_input)

title_selection = "Select a book"

if search_type == search_options[2]:
    book_titles = ["Select a book"] + list(tog_title_cover_image_path_map.keys())
    title_selection = st.selectbox("Select a book:", book_titles, on_change=clear_user_input, key="book_selection")

    if title_selection != "Select a book":
        with st.spinner("Loading book cover image..."):
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

#user_input = st.text_input("What would you like to know?", key="user_input", placeholder="Ask a question about the book or timeline arc...")
user_input = st.text_area("What would you like to know?", key="user_input", placeholder="Ask a question about the book or timeline arc...", height=100)
# Search button
search_clicked = st.button("🔍 Search")

# Output container
output_container = st.empty()

# Only run search if button is clicked
if search_clicked:
    if not user_input.strip():
        st.warning("Please enter a question before searching.")
    else:
        with st.spinner("Searching..."):
            if search_type == search_options[0] or search_type == search_options[1]:  # Timeline Arc or Deconstructive Analysis
                response = query_all_books(user_input, tog_book_title_vector_store_data_map)
            elif search_type == search_options[2] and title_selection != "Select a book":
                response = query_book(user_input, tog_book_title_vector_store_data_map[title_selection])
            else:
                response = None
                st.warning("Please select a book for book-specific search.")