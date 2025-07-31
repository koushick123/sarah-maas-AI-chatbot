import os.path
from threading import Thread

from langchain.chat_models import ChatOpenAI
from langchain.chains.summarize import load_summarize_chain
from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader, PdfWriter
from langchain.schema import Document
import os

llm = ChatOpenAI(model="gpt-4", temperature=0.1)

def summarize_with_langchain(text: str) -> str:
    # Use LangChain's summarization chain
    try:
        print(f"Size of context = {len(text)}")
        docs = [Document(page_content=text)]
        chain = load_summarize_chain(llm, chain_type="map_reduce")
        result = chain.run(docs)
    except Exception as openaiexcep:
        print(f"Exception from openAI = {openaiexcep}")

    return result.strip()

def extract_cover_page(pdf_path, outputfolderpath, filename, page_number=1):
    from PyPDF2 import PdfReader, PdfWriter
    import os
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    # Extract the specified page
    writer.add_page(reader.pages[page_number - 1])
    # Create the output directory if it doesn't exist
    output_dir = outputfolderpath + "/" + filename.removesuffix(".pdf")
    print("Check if "+ output_dir + " exists")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # write the page as a PNG file
    #image_file = output_dir + "/cover_page.png"
    image_file = os.path.join(output_dir, "cover_page.png")
    # Conver the the page to an image
    images = convert_from_path(pdf_path, first_page=page_number, last_page=page_number)
    images[0].save(image_file, 'PNG')
    print(f"Extracted cover page to {image_file}")

def extract_pdf_pages():
    print("Extracting PDF")
    reader = PdfReader(pdf_path)
    writer = PdfWriter()
    for page in range(5, 318):
        writer.add_page(reader.pages[page])

    with open(extracted_pdf_path, "wb") as output_pdf:
        writer.write(output_pdf)

    print(f"Extracted PDF to = {output_pdf}")

def extract_chapter_docs():
    if not os.path.exists(extracted_pdf_path):
        extract_pdf_pages()
    # Load the PDF document
    loader = PyPDFLoader(extracted_pdf_path)
    documents = loader.load()

    import re
    chapter_docs = []
    section_pattern = pattern = r"THE\s+ASSASSIN\s+AND\s+THE\s+([A-Z\s]+)"
    chapter_pattern = r"CHAPTER\n[0-9]+"
    section_title = ""
    chapter_title = ""
    for doc in documents:
        match = re.search(section_pattern, doc.page_content)
        if match:
            section_title_metadata = match.group().replace("\n"," ")
            section_title_metadata_cleaned = " ".join(section_title_metadata.split()[:-1])
            section_title = section_title_metadata_cleaned
        else:
            section_title_metadata_cleaned = section_title

        match = re.search(chapter_pattern, doc.page_content)
        if match:
            chapter_title_metadata = match.group().replace("\n"," ")
            chapter_title = chapter_title_metadata
        else:
            chapter_title_metadata = chapter_title

        doc.metadata = {
            "Section:":section_title_metadata_cleaned,
            "Chapter:":chapter_title_metadata,
            "Chapter Body/Passage:": doc.page_content
        }
        if chapter_title:
            chapter_docs.append(doc)

    return chapter_docs

CHAR_LIMIT = 8192

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
        section_name = doc.metadata["Section:"]

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
                all_page_contents += "\n".join(doc.page_content)

    chapter_summary = {}

    for sec_chap_name, page_content in chapter_page_content.items():
        if len(page_content) <= CHAR_LIMIT:
            print(f"Summarizing for {sec_chap_name}")
            chapter_summary[sec_chap_name] = summarize_with_langchain(page_content)
        else:
            # Extract CHAR_LIMIT tokens repeatedly until all done
            summaries = ""
            while True:
                context = page_content[:CHAR_LIMIT]
                summaries += summarize_with_langchain(context)

                # Exclude the first CHAR_LIMIT characters
                page_content = page_content[CHAR_LIMIT:]
                if len(page_content) <= CHAR_LIMIT:
                    break

            summaries += summarize_with_langchain(page_content)
            chapter_summary[sec_chap_name] = summaries

    for doc in chapter_documents:
        doc_chap_name = doc.metadata["Chapter:"]
        doc_section_name = doc.metadata["Section:"]
        doc.metadata["Summary"] = chapter_summary[doc_section_name+"-"+doc_chap_name]

    for doc in chapter_documents:
        print(doc.metadata)

def safe_summarize(text):
    while True:
        try:
            return summarize_with_langchain(text)
        except RateLimitError:
            print("Rate limit exceeded. Waiting 2 seconds before retrying...")
            time.sleep(2)

if __name__ == "__main__":
    # list all files from pdf_folder
    # read the above folder and extract cover page from each pdf file

    pdf_path = "/home/koushick/Young-Adult-ChatBot/Throne-Of-Glass/1-sarah-maas-assassins-blade.pdf"
    extracted_pdf_path = "/home/koushick/Young-Adult-ChatBot/Throne-Of-Glass/1-sarah-maas-assassins-blade/sm-assassins-blade-extracted.pdf"
    extract_chapter_summary()
    # pdf_folder = "/home/koushick/Young-Adult-ChatBot/Court-Of-Thorns-And-Roses"
    # for filename in os.listdir(pdf_folder):
    #     if filename.endswith(".pdf"):
    #         pdf_path = os.path.join(pdf_folder, filename)
    #         print(f"Extracting cover page from {pdf_path}")
    #         extract_cover_page(pdf_path, pdf_folder, filename, page_number=1)
    #     else:
    #         print(f"Skipping {filename}, not a PDF file.")
    # print("Cover page extraction completed.")