import os
from pdf2image import convert_from_path
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


if __name__ == "__main__":
    # list all files from pdf_folder
    # read the above folder and extract cover page from each pdf file
    
    pdf_folder = "/home/koushick/Young-Adult-ChatBot/Court-Of-Thorns-And-Roses"
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Extracting cover page from {pdf_path}")
            extract_cover_page(pdf_path, pdf_folder, filename, page_number=1)
        else:
            print(f"Skipping {filename}, not a PDF file.")
    print("Cover page extraction completed.")