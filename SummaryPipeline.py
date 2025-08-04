import os.path
import time
from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader, PdfWriter
from langchain.schema import Document
import os

llm = ChatOpenAI(model="chatgpt-4o-latest", temperature=0.1)

def summarize_with_langchain(text: str) -> str:
    # Use LangChain's summarization chain
    print(f"Size of context = {len(text)}")
    docs = [Document(page_content=text)]
    chain = load_summarize_chain(llm, chain_type="stuff")
    result = chain.run(docs)
    time.sleep(61)
    return result.strip()

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

CHAR_LIMIT = 500000

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

def extract_chapter_summary():
    chapter_documents = extract_chapter_docs()
    # Create a Dict of Chapter-Name: Summary of all chapter pages - Use OpenAI API to get summary
    # Loop through original docs and for entry matching Chapter-Name , retrieve the summary from Dict
    # and update metadata as 'summary'
    chapter_page_content = {}
    all_page_contents = None
    old_chapter_name = None
    key = None
    for doc in chapter_documents:
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
            all_page_contents = doc.page_content
        else:
            if all_page_contents is None:
                all_page_contents = doc.page_content
            else:
                all_page_contents += "\n" + doc.page_content

    # Populate the page contents of last chapter
    chapter_page_content[key] = all_page_contents

    filtered_dict = {
        k: v for k, v in chapter_page_content.items() if "Chapter" in k
    }
    sorted_dict = dict(sorted(filtered_dict.items(), key=lambda item: len(item[1])))
    print(len(sorted_dict.items()))
    for key, value in sorted_dict.items():
        print(f"Part = {key}")
    #     print(f"Page content = {value}")

    # for sec_chap_name, page_content in chapter_page_content.items():
    #     print(f"Section = {sec_chap_name}\n")
    #     print(f"Page Content = {len(page_content)}\n")
    #     if len(page_content) <= CHAR_LIMIT:
    #         print(f"Summarizing for {sec_chap_name}")
    #         chapter_summary[sec_chap_name] = summarize_with_langchain(page_content)
    #     else:
    #         # Extract CHAR_LIMIT tokens repeatedly until all done
    #         summaries = ""
    #         while True:
    #             context = page_content[:CHAR_LIMIT]
    #             print(f"Summarizing for {sec_chap_name} after truncating to context window size")
    #             print(f"Sending context size = {len(context)}")
    #             summaries += summarize_with_langchain(context)
    #
    #             # Exclude the first CHAR_LIMIT characters
    #             page_content = page_content[CHAR_LIMIT:]
    #             if len(page_content) <= CHAR_LIMIT:
    #                 break
    #
    #         summaries += summarize_with_langchain(page_content)
    #         print(f"Summary for {sec_chap_name} ==== {summaries} ")
    #         chapter_summary[sec_chap_name] = summaries
    #
    # for doc in chapter_documents:
    #     doc_chap_name = doc.metadata["Chapter:"]
    #     doc_section_name = doc.metadata["Section:"]
    #     doc.metadata["Summary"] = chapter_summary[doc_section_name+"-"+doc_chap_name]
    #
    # for doc in chapter_documents:
    #     print(doc.metadata)


if __name__ == "__main__":
    pdf_path = "/home/koushick/Young-Adult-ChatBot/Crescent-City/Maas_Sarah_J_-_Crescent_City_1_-_House_of_Earth_and_Blood.pdf"
    extracted_pdf_path = "/home/koushick/Young-Adult-ChatBot/Crescent-City/1-sm-cc-earth-and-blood-extracted.pdf"
    extract_chapter_summary()