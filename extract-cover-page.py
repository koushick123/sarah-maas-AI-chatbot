import os.path

from pdf2image import convert_from_path
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader, PdfWriter

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

    print(len(chapter_docs))
    # List the doc contents
    count=0
    section1_docs = []
    for chap_doc in chapter_docs:
        section_str= chap_doc.metadata['Section:']
        if "THE PIRATE" in section_str:
            section1_docs.append(chap_doc)

    print(section1_docs)

if __name__ == "__main__":
    # list all files from pdf_folder
    # read the above folder and extract cover page from each pdf file

    pdf_path = "/home/koushick/Young-Adult-ChatBot/Throne-Of-Glass/1-sarah-maas-assassins-blade.pdf"
    extracted_pdf_path = "/home/koushick/Young-Adult-ChatBot/Throne-Of-Glass/1-sarah-maas-assassins-blade/sm-assassins-blade-extracted.pdf"
    extract_chapter_docs()
    # pdf_folder = "/home/koushick/Young-Adult-ChatBot/Court-Of-Thorns-And-Roses"
    # for filename in os.listdir(pdf_folder):
    #     if filename.endswith(".pdf"):
    #         pdf_path = os.path.join(pdf_folder, filename)
    #         print(f"Extracting cover page from {pdf_path}")
    #         extract_cover_page(pdf_path, pdf_folder, filename, page_number=1)
    #     else:
    #         print(f"Skipping {filename}, not a PDF file.")
    # print("Cover page extraction completed.")