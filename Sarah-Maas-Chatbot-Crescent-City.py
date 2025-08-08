import base64
import streamlit as st
from tinydb import TinyDB, Query
from streamlit_modal import Modal
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain_community.llms import Ollama
from langchain.chains import LLMChain
from fastapi import FastAPI

# Initialize FastAPI app
app = FastAPI()

@app.get("/healtcheck")
def healthcheck():
    """
    Health check endpoint to verify the API is running.
    """
    return {"status": "ok", "message": "Sarah Maas AI Chatbot is running!"}

# def get_base64_image(image_path):
#     with open(image_path, "rb") as img_file:
#         print(f"Base64 encoding image {image_path}.")
#         base64_image = base64.b64encode(img_file.read()).decode()
#         return base64_image
#
# def summarize_with_gpt4turbo(context_chapter_summary):
#     system_message = (
#         "You are a knowledgeable literary research assistant with deep familiarity "
#         "with Sarah J. Maas's *Crescent City* series of books.\n"
#         "Focus on offering thoughtful, research-level insight into the text.\n"
#         "Avoid generic filler and do not add introductory phrases."
#     )
#
#     chat_prompt = ChatPromptTemplate.from_messages([
#         SystemMessagePromptTemplate.from_template(system_message),
#         HumanMessagePromptTemplate.from_template(
#             "Answer the following question using ONLY the context provided.\n"
#             "Context:\n{context}\n\n"
#             "Question: {question}\n\n"
#             "Answer:"
#         ),
#     ])
#
#     temperature = 0.1
#
#     selected_llm = Ollama(
#         model="gpt-4-turbo",
#         temperature=temperature
#         # callbacks=[stream_handler]
#     )
#
#     chain = LLMChain(llm=selected_llm, prompt=chat_prompt)
#     return chain.run(context=context_chapter_summary, question="Summarize the chapter in detail, focusing on characterisation and plot progression")
#
#
# def fetch_book_contents(sel_chap):
#     # Get the chapter contents as per selection
#     Chapter = Query()
#     page_contents = crescent_city_db.get(Chapter.Name == sel_chap)["Page Content"]
#     # Create a unique key for modal visibility control
#     modal_key = f"{sel_chap}_modal_visible"
#     # Initialize session state
#     if modal_key not in st.session_state:
#         st.session_state[modal_key] = True
#
#     modal = Modal(
#         sel_chap,
#         key=sel_chap+"-key",
#
#         # Optional
#         padding=20,  # default value
#         max_width=800  # default value
#     )
#
#     # Only show if modal visibility is True
#     if st.session_state[modal_key]:
#         with modal.container():
#             # Display content with limited height so it scrolls
#             st.markdown(
#                 f"""
#                     <div style='text-align: left;'>
#                     {page_contents}
#                     </div>
#                 """, unsafe_allow_html=True
#             )
#             if st.button("❌ Close", key="close-modal"):
#                 st.session_state.modal_open = False
#
# st.set_page_config(page_title="Sarah Maas AI Chatbot", layout="centered")
# img_base64 = get_base64_image("SarAIh_Maas_V2.png")
# st.markdown(
#     f"""
#         <div style='text-align: center;'>
#             <img src="data:image/png;base64,{img_base64}" width="300"><br>
#         </div>
#         """,
#     unsafe_allow_html=True
# )
#
# st.markdown(f"""
# <div style='text-align: center;'>
#     <h5>SarAIh Maas can give you detailed analysis on the Crescent City series of Sarah Maas Books</h5>
# </div>
# """, unsafe_allow_html=True)
#
# # List the drop down of Crescent city books
# book_titles = ["Select a Book",
#                "Crescent City - House of Earth and Blood",
#                "Crescent City - House of Sky and Breath",
#                "Crescent City - House of Flame and Shadow"]
# selected_book = st.selectbox("Select a book", book_titles, key="book_selection")
#
# selected_chapter = None
# crescent_city_db = None
#
# if selected_book != book_titles[0]:
#     chapter_titles = ["Select a Chapter"]
#     if selected_book == book_titles[1]:
#         crescent_city_db = TinyDB('sm-crescent-city-book-1.json')
#         all_docs = crescent_city_db.all()
#         for doc in all_docs:
#             chapter_titles.append(doc['Name'])
#
#     selected_chapter = st.selectbox("Choose a Chapter", chapter_titles, key="chapter_selection")
#
# # --- Summary Options Layout ---
# if selected_chapter and selected_chapter != "Select a Chapter":
#     st.markdown("---")
#     summary_col1, summary_col2, summary_col3 = st.columns(3)
#
#     with summary_col2:
#         st.markdown("""
#             <div style='
#                 font-size: 22px;
#                 text-align: center;
#                 white-space: nowrap;
#                 overflow: hidden;
#             '>
#                 <strong>Choose Summary Type</strong>
#             </div>
#         """, unsafe_allow_html=True)
#
#     st.markdown("<br>",unsafe_allow_html=True)
#
#     st.markdown("<h4 style='font-size: 18px;'>🧪 Deep Analysis Summaries</h4>", unsafe_allow_html=True)
#
#     deep_summary_choice = st.radio("Choose Deep Summary Option", [
#         "Summary 1 - Summarize entire chapter using regular ChatGPT",
#         "Summary 2 - Summarize chapter part by part and merge",
#         "Summary 3 - Merge Summary 1 and Summary 2 using regular ChatGPT"
#     ], key="deep_summary")
#
#     button_col1, button_col2, button_col3 = st.columns(3)
#
#     with button_col2:
#         deep_summary_button = st.button("Generate Summary", key="deep_summary_button")
#         if deep_summary_choice and deep_summary_choice != "Select a Summary":
#             st.markdown(f"<h5 style='font-size: 16px;'>Selected: {deep_summary_choice}</h5>", unsafe_allow_html=True)
#         else:
#             st.error("No Deep Summary Selected", icon="🚫")
#             # st.markdown("<h5 style='font-size: 16px;'>No Deep Summary Selected</h5>", unsafe_allow_html=True)
#
#     # Show chapter button (optional after choices)
#     st.markdown("---")
