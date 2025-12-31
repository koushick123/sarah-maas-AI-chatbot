import asyncio
import json
import os
import re
import shutil
import tempfile

import fitz
from PyPDF2 import PdfReader
from bson import ObjectId

# from SarahMaasAzureOCR import extract_and_save_text_from_ocr_page
from SarahMaasSearchChapterHeadings import filter_chapter_headings_for_chapter_beginning
from Sarah_Maas_Chatbot_Crescent_City import collection_book_chunk_metadata, collection_book_staging, \
    collection_book_chunks, fs


async def book_analyzer_events(book_id: str):
    yield f"data: Starting analysis for book_id: {book_id}\n\n"
    await asyncio.sleep(0.5)

    # Check if chunk metadata exists
    yield f"data: Checking chunk metadata for book_id: {book_id}\n\n"
    await asyncio.sleep(0.5)

    try:
        book_doc = collection_book_chunk_metadata.find_one({"file_id": book_id})
        if book_doc:
            number_of_chunks = book_doc["total_chunks"]
            yield f"data: Chunk metadata found. Total chunks: {number_of_chunks} for book_id: {book_id}\n\n"
        else:
            yield f"data: Chunk metadata not found for book_id: {book_id}. Checking staging record...\n\n"
            print("Book metadata not found for book_id: ", book_id)
            print("Check for staging record for the book_id: ", book_id)
            staging_doc = collection_book_staging.find_one({"book_id": book_id})
            if staging_doc:
                print("Staging record found, now start chunking...")
                yield f"data: Staging record found. Starting chunking process for book_id: {book_id}\n\n"
                print(f"🔧 TOOL CALLED: get_book_chunks with book_id={book_id}")


                # Step 1: Check if chunks already exist
                existing_chunks = check_existing_chunks(book_id)
                if existing_chunks:
                    existing_chunks["success"] = True
                    yield f"data: Existing chunks check completed for book_id: {book_id} and found {len(existing_chunks)} chunks\n\n"
                    yield f"data: {json.dumps(existing_chunks)}\n\n"
                else:

                    yield f"data: No chunks found for book_id: {book_id}. Preparing to chunk the book.\n\n"
                    # Step 2: Fetch PDF from GridFS
                    pdf_binary = fetch_pdf_from_gridfs(book_id)

                    yield f"data: PDF fetched from GridFS for book_id: {book_id}, size: {len(pdf_binary)} bytes\n\n"
                    # Step 3: Create temporary PDF file
                    temp_pdf_path = create_temp_pdf(pdf_binary)

                    yield f"data: Temporary PDF created at {temp_pdf_path} for book_id: {book_id}\n\n"
                    # Step 4: Find first chapter page
                    first_page_num, page_count = find_first_chapter_page(temp_pdf_path)
                    print(f"First chapter starts on page: {first_page_num}")
                    if first_page_num == -1:
                        book_data = create_book_metadata(
                            book_id, page_count, first_page_num, [], "",
                            "No chapter or part found"
                        )
                        yield f"data: No chapter or part found in the book  {json.dumps(book_data)}\n\n"
                    else:
                        print(f"📖 Extracting text from page {first_page_num} to {page_count}...")

                        yield f"data: Extracting text from page {first_page_num} to {page_count} for book_id: {book_id}\n\n"
                        # Step 5: Extract text from pages
                        full_text, map_page_num_content = extract_text_from_pages(
                            temp_pdf_path, first_page_num, page_count
                        )

                        yield f"data: Text extraction completed for book_id: {book_id}, total characters: {len(full_text)}\n\n"
                        # Step 6: Process chapters
                        final_chapters = process_chapters_from_text(full_text)
                        error_msg = None

                        yield f"data: Chapter processing completed for book_id: {book_id}, total chapters: {len(final_chapters)}\n\n"
                        # Step 7: If no chapters found, try OCR
                        if not final_chapters:
                            print("No chapters found. Attempting OCR processing...")
                            final_chapters, error_msg = process_with_ocr(
                                temp_pdf_path, book_id, first_page_num, map_page_num_content, ocr_page_count=1
                            )

                            if final_chapters:
                                # Rebuild full_text from chapters
                                full_text = " ".join(final_chapters)

                                # Step 8: Create and return metadata
                                yield f"data: OCR processing succeeded for book_id: {book_id}, total chapters: {len(final_chapters)}\n\n"
                                book_doc = create_book_metadata(
                                    book_id, page_count, first_page_num,
                                    final_chapters, full_text, error_msg
                                )

                                if not book_doc["success"]:
                                    print("Error in processing book chunks: ", book_doc.get("error", "Unknown error"))
                                    yield f"data: Error in chunking process: {json.dumps(book_doc)}\n\n"
                                else:
                                    number_of_chunks = book_doc["total_chunks"]
                                    yield f"data: Chunking completed. Total chunks: {number_of_chunks} for book_id: {book_id}\n\n"
                            else:
                                yield f"data: OCR processing failed for book_id: {book_id}. {error_msg}\n\n"
                        else:
                            book_doc = create_book_metadata(
                                book_id, page_count, first_page_num,
                                final_chapters, full_text
                            )
                            number_of_chunks = book_doc["total_chunks"]
                            yield f"data: Chunking completed. Total chunks: {number_of_chunks} for book_id: {book_id}\n\n"
            else:
                number_of_chunks = 0
                print(f"Chunk count for book_id {book_id} is {number_of_chunks}")
                yield f"data: No staging record found for book_id: {book_id}. Cannot proceed with analysis.\n\n"

    except Exception as e:
        print(f"❌ Error in get_book_chunks tool: {str(e)}")
        book_doc = {
            "error": str(e),
            "book_id": book_id,
            "success": False
        }
        yield f"data: Error in chunking process: {json.dumps(book_doc)}\n\n"

    finally:
        # Clean up temporary file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)


def check_existing_chunks(book_id: str):
    """Check if chunks already exist for the book."""
    chunks_doc = collection_book_chunks.find_one({"book_id": book_id})
    if chunks_doc:
        print(f"Book chunks already exist for book_id: {book_id}")
        chunks_dict = dict(chunks_doc)
        # Convert ObjectId fields to strings
        for key, value in chunks_dict.items():
            if isinstance(value, ObjectId):
                chunks_dict[key] = str(value)
        return chunks_dict
    return None

def fetch_pdf_from_gridfs(book_id: str):
    """Retrieve PDF binary from GridFS."""
    file_id = ObjectId(collection_book_staging.find_one({"book_id": book_id})["file_id"])
    pdf_file = fs.get(file_id)
    pdf_binary = pdf_file.read()
    print(f"📄 PDF fetched, size: {len(pdf_binary)} bytes")
    return pdf_binary

def create_temp_pdf(pdf_binary: bytes):
    """Create a temporary PDF file from binary data."""
    temp_pdf = tempfile.NamedTemporaryFile(delete=False, suffix='.pdf')
    temp_pdf.write(pdf_binary)
    temp_pdf.close()
    return temp_pdf.name

def find_first_chapter_page(pdf_path: str):
    """Find the page number where the first chapter begins."""
    with open(pdf_path, "rb") as pdf_document:
        reader = PdfReader(pdf_document)
        page_count = len(reader.pages)
        print("Extract text from PDF to find first chapter...")

        for page_num in range(page_count):
            page = reader.pages[page_num]
            print(f"Page number: {page_num + 1}")
            page_text = page.extract_text() or ""
            print(f"Page Text ====== {page_text[:50]}")

            if does_part_or_chapter_exist_only_once(page_text) and find_first_chapter_or_part(page_text.strip()):
                return page_num + 1, page_count

        return -1, page_count


def does_part_or_chapter_exist_only_once(page_text: str) -> bool:
    # Check for Part
    firstindex, lastindex = get_first_and_last_index_of_part_or_chapter(page_text, "part")
    if firstindex != -1 and lastindex != -1:
        # Check for Chapter if it repeats (In case of CONTENTS page)
        firstindexchapter, lastindexchapter = get_first_and_last_index_of_part_or_chapter(page_text, "chapter")
        if firstindexchapter != -1 and lastindexchapter != -1:
            return firstindexchapter == lastindexchapter
        # Chapter does not exist, check if part exists only once
        return firstindex == lastindex
    else:
        # Check for Chapter if Part doesn't exist
        firstindex, lastindex = get_first_and_last_index_of_part_or_chapter(page_text, "chapter")
        if firstindex != -1 and lastindex != -1:
            # Check for Part if it repeats (In case of CONTENTS page)
            firstindexpart, lastindexpart = get_first_and_last_index_of_part_or_chapter(page_text, "part")
            if firstindexpart != -1 and lastindexpart != -1:
                return firstindexpart == lastindexpart
            # Part does not exist, check if chapter exists only once
            return firstindex == lastindex
    # Neither Chapter nor Part exists
    return False


def find_first_chapter_or_part(text: str):
    # Patterns for "Part", "Chapter" as independent words at line start, number, or Roman numeral
    patterns = [
        (r"^\s*\bPart\b(?!\w)", "Part"),
        (r"^\s*\bChapter\b(?!\w)", "Chapter"),
        (r"^\s*\d+\b", "Number"),
        (r"^\s*[IVXLCDM]+\b", "RomanNumeral")
    ]
    for pattern, label in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.start() == 0
    return False


def get_first_and_last_index_of_part_or_chapter(page_text: str, part_or_chapter: str) -> tuple[int, int]:
    # Include a check for 'Section' as well during 'Part' check
    firstindex = page_text.lower().find(part_or_chapter.lower())
    print(f"First index of {part_or_chapter} = {firstindex}")
    lastindex = page_text.lower().rfind(part_or_chapter.lower())
    print(f"Last index of {part_or_chapter} = {lastindex}")

    if part_or_chapter.lower() == "part":
        if firstindex == -1 and lastindex == -1:
            # Check for 'Section' if 'Part' not found
            firstindex = page_text.lower().find("section")
            print(f"First index of Section = {firstindex}")
            lastindex = page_text.lower().rfind("section")
            print(f"Last index of Section = {lastindex}")

    return firstindex, lastindex


def create_book_metadata(book_id: str, page_count: int, first_page_num: int,
                         final_chapters: list, full_text: str, error_msg: str = None):
    """Create metadata dictionary for the book."""
    metadata = {
        "file_id": book_id,
        "file_name": f"{book_id}.pdf",
        "page_count": page_count,
        "page_for_first_chapter": first_page_num,
        "total_chunks": len(final_chapters) if final_chapters else 0,
        "total_characters": len(full_text) if full_text else 0,
        "success": error_msg is None
    }

    if error_msg:
        metadata["error"] = error_msg

    # Convert ObjectId to strings
    for key, value in metadata.items():
        if isinstance(value, ObjectId):
            metadata[key] = str(value)

    return metadata


def extract_text_from_pages(pdf_path: str, first_page_num: int, page_count: int):
    """Extract and clean text from specified page range."""
    with open(pdf_path, "rb") as pdf_document:
        full_text = ""
        map_page_num_content = {}
        reader = PdfReader(pdf_document)
        page_list = reader.pages

        for page_num in range(first_page_num, page_count):
            page = page_list[page_num - 1]
            page_content = clean_text(page.extract_text())
            full_text += page_content + " "
            map_page_num_content[page_num] = page_content

        return full_text, map_page_num_content


def clean_text(page_text):
    # Remove page numbers (common patterns)
    page_text = re.sub(r"^\s*\d+\s*$", "", page_text, flags=re.MULTILINE)

    # Remove headers
    page_text = re.sub(r"^\s*.*?Book.*?$", "", page_text, flags=re.MULTILINE)

    # Fix hyphenated line breaks
    page_text = re.sub(r"-\n", "", page_text)

    # Fix normal line breaks
    page_text = re.sub(r"\n+", "\n", page_text)

    return page_text.strip()


def process_chapters_from_text(full_text: str):
    """Split text into chapters and handle long chapters."""
    chapters = split_into_chapters(full_text)
    print(f"Total chapters extracted: {len(chapters)}")

    if chapters:
        return split_long_chapters(chapters, 8000)
    return []


def split_into_chapters(full_text):
    chapter_regex = re.compile(
        r"chapter[\n\r]?[\s]*[\d]*",
        re.MULTILINE | re.IGNORECASE
    )

    number_regex = re.compile(r"\d")

    chapters = []
    matches = list(chapter_regex.finditer(full_text))
    print(f"Total chapter matches found: {len(matches)}")

    if not matches:
        matches = list(number_regex.finditer(full_text))
        print(f"Total chapter number matches found: {len(matches)}")

    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)

        chapter_number = i + 1
        chapter_text = full_text[start:end].strip()

        print(f"Chapter number {chapter_number}")
        print(f"Chapter text {chapter_text[:20]}")
        chapters.append({
            "chapter": chapter_number,
            "text": chapter_text
        })

    return chapters


def split_long_chapters(chapters, max_chars=8000):
    final_chunks = []
    page_number = 1

    for ch in chapters:
        text = ch["text"]

        if len(text) <= max_chars:
            final_chunks.append({
                "chapter": ch["chapter"],
                "part": None,
                "text": text,
                "page_number": page_number
            })
        else:
            # split
            parts = [text[i:i + max_chars] for i in range(0, len(text), max_chars)]
            labels = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

            for idx, part_text in enumerate(parts):
                final_chunks.append({
                    "chapter": ch["chapter"],
                    "part": labels[idx],
                    "text": part_text,
                    "page_number": page_number
                })
                if idx < len(parts) - 1:
                    page_number += 1

        page_number += 1

    return final_chunks


def process_with_ocr(pdf_path: str, book_id: str, first_page_num: int, map_page_num_content: dict,
                     ocr_page_count: int = 1) -> tuple:
    """Process pages using OCR when regular text extraction fails."""
    full_text = ""

    # Convert pages to images
    page_nums = list(map_page_num_content.keys())[:ocr_page_count]
    convert_pages_to_images(pdf_path, book_id, page_nums, ocr_page_count)

    # Extract text from OCR
    # extract_and_save_text_from_ocr_page(first_page_num, first_page_num + ocr_page_count, book_id)

    # Filter chapter headings
    map_page_num_page_heading = filter_chapter_headings_for_chapter_beginning(
        first_page_num, first_page_num + ocr_page_count
    )

    if not map_page_num_page_heading:
        print("Unable to extract text using OCR.")
        return None, "Unable to extract text using OCR."

    # Get non-blank chapter headings
    map_page_num_chapter_heading = {
        page_no: page_heading
        for page_no, page_heading in map_page_num_page_heading.items()
        if page_heading
    }

    if not map_page_num_chapter_heading:
        print("Chapter headings are blank. Unable to proceed.")
        return None, "Chapter headings are blank. Unable to proceed."

    # Add chapter prefixes
    chapter_no = 1
    for page_no in map_page_num_chapter_heading:
        page_text = map_page_num_content[page_no]
        page_prefix = f"Chapter {chapter_no}\n\n"
        page_text = page_prefix + page_text
        chapter_no += 1
        page_text = clean_text(page_text)
        full_text += page_text + " "

    # Reprocess to split by chapters
    chapters = split_into_chapters(full_text)
    print(f"Total chapters extracted after OCR: {len(chapters)}")
    final_chapters = split_long_chapters(chapters, 8000)

    # Clean up
    cleanup_ocr_files()

    return final_chapters, None


def cleanup_ocr_files():
    """Clean up temporary OCR files."""
    try:
        if os.path.exists("pages_for_OCR"):
            shutil.rmtree("pages_for_OCR")
        if os.path.exists("map_of_page_nos_chapter_heading.json"):
            os.remove("map_of_page_nos_chapter_heading.json")
        print("Clean up completed")
    except Exception as e:
        print(f"Error during cleanup: {e}")


def convert_pages_to_images(pdf_path: str, book_id: str, page_nums: list, max_pages: int = 1):
    """Convert PDF pages to images for OCR processing."""
    pdf_doc = fitz.open(pdf_path)
    os.makedirs("pages_for_OCR", exist_ok=True)

    count = 0
    for page_num in page_nums:
        if count >= max_pages:
            break

        output_path = f"pages_for_OCR/page_{book_id}_{page_num}.jpg"

        if os.path.exists(output_path):
            count += 1
            continue

        # Convert to image
        fitz_page = pdf_doc[page_num - 1]
        pix = fitz_page.get_pixmap(dpi=300)
        pix.save(output_path)
        print(f"✓ Created: {output_path}")
        count += 1

    pdf_doc.close()
    return count