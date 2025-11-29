import fitz  # PyMuPDF
import os
from pathlib import Path

def convert_pdf_to_images(pdf_path, output_dir="output_images"):
    """
    Reads a PDF file and saves each page as a high-resolution image using fitz.
    
    Args:
        pdf_path (str or Path): The path to the input PDF file.
        output_dir (str or Path): The directory where image files will be saved.
    """
    pdf_path = Path(pdf_path)
    output_dir = Path(output_dir)

    # Ensure the output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Saving images to: {output_dir}")

    try:
        # Open the PDF document
        with fitz.open(pdf_path) as pdf_doc:
            print(f"Processing '{pdf_path}' ({pdf_doc.page_count} pages)...")

            for page_num in range(1, pdf_doc.page_count + 1):
                # Your logic: Access the page (page_num - 1 because it's 0-indexed)
                fitz_page = pdf_doc[page_num - 1]
                
                # Get a high-resolution pixmap (image representation)
                pix = fitz_page.get_pixmap(dpi=300)
                
                # Define the output file path (e.g., invoice_page_001.png)
                output_filename = f"{pdf_path.stem}_page_{page_num:03d}.png"
                output_path = output_dir / output_filename
                
                # Save the pixmap to a file (PNG format by default)
                pix.save(output_path)
                
                print(f"✓ Created: {output_path}")

        print("\nConversion complete!")

    except FileNotFoundError:
        print(f"Error: The file '{pdf_path}' was not found.")
    except Exception as e:
        print(f"An error occurred: {e}")

# --- Example Usage ---
if __name__ == "__main__":
    # Replace 'input_document.pdf' with the path to your actual PDF file
    input_pdf_file = "input_document.pdf" 
    convert_pdf_to_images(input_pdf_file, output_dir="pdf_output_images")

