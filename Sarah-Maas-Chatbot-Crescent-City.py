import base64
import streamlit as st
from tinydb import TinyDB, Query
from streamlit_modal import Modal

def get_base64_image(image_path):
    with open(image_path, "rb") as img_file:
        print(f"Base64 encoding image {image_path}.")
        base64_image = base64.b64encode(img_file.read()).decode()
        return base64_image

def fetch_book_contents(sel_chap):
    # Get the chapter contents as per selection
    Chapter = Query()
    page_contents = crescent_city_db.get(Chapter.Name == sel_chap)["Page Content"]
    # Create a unique key for modal visibility control
    modal_key = f"{sel_chap}_modal_visible"
    # Initialize session state
    if modal_key not in st.session_state:
        st.session_state[modal_key] = True

    modal = Modal(
        sel_chap,
        key=sel_chap+"-key",

        # Optional
        padding=20,  # default value
        max_width=800  # default value
    )

    # Only show if modal visibility is True
    if st.session_state[modal_key]:
        with modal.container():
            # Display content with limited height so it scrolls
            with st.container():
                st.markdown(
                    f"""
                        <div style='text-align: left;'>
                        {page_contents}
                        </div>
                    """, unsafe_allow_html=True
                )
                if st.button("❌ Close", key="close-modal"):
                    st.session_state.modal_open = False

st.set_page_config(page_title="Sarah Maas AI Chatbot", layout="centered")
img_base64 = get_base64_image("SarAIh_Maas_V2.png")
st.markdown(
    f"""
        <div style='text-align: center;'>
            <img src="data:image/png;base64,{img_base64}" width="300"><br>
        </div>
        """,
    unsafe_allow_html=True
)

st.markdown(f"""
<div style='text-align: center;'>
    <h5>SarAIh Maas can give you detailed analysis on the Crescent City series of Sarah Maas Books</h5>
</div>
""", unsafe_allow_html=True)

# List the drop down of Crescent city books
book_titles = ["Select a Book","Crescent City - House of Earth and Blood","Crescent City - House of Sky and Breath","Crescent City - House of Flame and Shadow"]
selected_book = st.selectbox("Select a book", book_titles, key="book_selection")

if selected_book != book_titles[0]:
    # Fetch chapter summaries
    chapter_titles = ["Select a Chapter"]
    if selected_book == book_titles[1]:
        crescent_city_db = TinyDB('sm-crescent-city-book-1.json')
        all_docs = crescent_city_db.all()
        for doc in all_docs:
            chapter_titles.append(doc['Name'])

    selected_chapter = st.selectbox("Select a Chapter",chapter_titles, key="chapter_selection")
    if selected_chapter != "Select a Chapter":
        select_button = st.button("Show Chapter from Book")
        if select_button:
            fetch_book_contents(selected_chapter)


# Option for Deeper Analysis Summary
# Summary 1 - from gpt-4-turbo

# Summary 2 - from langchain map_reduce
# Final summary - from Summary 1 and Summary 2 via gpt-4-turbo

# Option for general summary
# Final summary - from gpt-4-turbo