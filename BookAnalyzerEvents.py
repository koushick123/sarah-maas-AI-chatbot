import asyncio
import json
import os
import re
import tempfile
import urllib.parse

import fitz
from PyPDF2 import PdfReader
from bson import ObjectId
from pymongo import MongoClient

from SarahMaasChatbotCrescentCity import decrypt_mongo_user, decrypt_mongo_password, decrypt_mongo_hosturl
from SarahMaasSearchChapterHeadings import filter_chapter_headings_for_chapter_beginning
from kafka import KafkaProducer

username = urllib.parse.quote_plus(decrypt_mongo_user())
password = urllib.parse.quote_plus(decrypt_mongo_password())
host_url = decrypt_mongo_hosturl()
uri = f"mongodb+srv://{username}:{password}@{host_url}/?retryWrites=true&w=majority&appName=dev-cluster"

client = MongoClient(uri)
db = client["sarah-maas-db"]
sm_map_page_nos_chap_heading_collection = db['sarah-maas-map-page-nos-chapter-heading']
book_chunk_metadata_collection = db["sarah-maas-books-metadata"]
book_chunk_collection = db["sarah-maas-books-chunk"]

# Create producer
producer = KafkaProducer(
    bootstrap_servers=['localhost:9094'],
    value_serializer=lambda v: json.dumps(v).encode('utf-8')
)

topic = 'mytopic'

# from SarahMaasAzureOCR import extract_and_save_text_from_ocr_page
from SarahMaasChatbotCrescentCity import collection_book_chunk_metadata, collection_book_staging, \
    collection_book_chunks, fs


async def book_analyzer_events(book_id: str):
    yield f"data: Starting analysis for book_id: {book_id}\n\n"
    await asyncio.sleep(0.5)
    # Check if chunk metadata exists
    yield f"data: Checking chunk metadata for book_id: {book_id}\n\n"
    await asyncio.sleep(0.5)

    temp_pdf_path = ""
    try:
        book_doc = collection_book_chunk_metadata.find_one({"file_id": book_id})
        await asyncio.sleep(0.5)
        if book_doc:
            number_of_chunks = book_doc["total_chunks"]
            yield f"data: Chunk metadata found. Total chunks: {number_of_chunks} for book_id: {book_id}\n\n"
            await asyncio.sleep(0.5)
        else:
            yield f"data: Chunk metadata not found for book_id: {book_id}. Checking staging record...\n\n"
            await asyncio.sleep(0.5)
            print("Book metadata not found for book_id: ", book_id)
            print("Check for staging record for the book_id: ", book_id)
            staging_doc = collection_book_staging.find_one({"book_id": book_id})
            await asyncio.sleep(0.5)
            if staging_doc:
                print("Staging record found, now start chunking...")
                yield f"data: Staging record found. Starting chunking process for book_id: {book_id}\n\n"
                await asyncio.sleep(0.5)
                print(f"🔧 TOOL CALLED: get_book_chunks with book_id={book_id}")

                # Step 1: Check if chunks already exist
                existing_chunks = check_existing_chunks(book_id)
                if existing_chunks:
                    existing_chunks["success"] = True
                    yield f"data: Existing chunks check completed for book_id: {book_id} and found {len(existing_chunks)} chunks\n\n"
                    yield f"data: {json.dumps(existing_chunks)}\n\n"
                    await asyncio.sleep(0.5)
                else:
                    yield f"data: No chunks found for book_id: {book_id}. Preparing to chunk the book.\n\n"
                    await asyncio.sleep(0.5)
                    # Step 2: Fetch PDF from GridFS
                    pdf_binary = fetch_pdf_from_gridfs(book_id)

                    yield f"data: PDF fetched from GridFS for book_id: {book_id}, size: {len(pdf_binary)} bytes\n\n"
                    await asyncio.sleep(0.5)
                    # Step 3: Create temporary PDF file
                    temp_pdf_path = create_temp_pdf(pdf_binary)

                    yield f"data: Temporary PDF created at {temp_pdf_path} for book_id: {book_id}\n\n"
                    await asyncio.sleep(0.5)
                    # Step 4: Find first chapter page
                    first_page_num, end_page_num = find_first_chapter_page(temp_pdf_path)
                    print(f"First chapter starts on page: {first_page_num}")
                    if first_page_num == -1:
                        book_data = create_book_metadata(
                            book_id, end_page_num, first_page_num, [], "",
                            "No chapter or part found"
                        )
                        yield f"data: No chapter or part found in the book  {json.dumps(book_data)}\n\n"
                    else:
                        print(f"📖 Extracting text from page {first_page_num} to {end_page_num}...")

                        yield f"data: Extracting text from page {first_page_num} to {end_page_num} for book_id: {book_id}\n\n"
                        await asyncio.sleep(0.5)
                        # Step 5: Extract text from pages
                        full_text, map_page_num_content = extract_text_from_pages(
                            temp_pdf_path, first_page_num, end_page_num
                        )

                        yield f"data: Text extraction completed for book_id: {book_id}, total characters: {len(full_text)}\n\n"
                        await asyncio.sleep(0.5)
                        # Step 6: Process chapters
                        final_chapters = process_chapters_from_text(full_text, book_id)
                        error_msg = None

                        yield f"data: Chapter processing completed for book_id: {book_id}, total chapters: {len(final_chapters)}\n\n"
                        await asyncio.sleep(0.5)
                        # Step 7: If no chapters found, try OCR
                        if not final_chapters:
                            print("No chapters found. Attempting OCR processing...")
                            yield f"data: No chapters found. Attempting OCR processing..\n\n"
                            await asyncio.sleep(0.5)
                            page_count = (end_page_num-first_page_num)
                            ocr_generator = process_with_ocr(temp_pdf_path, book_id, first_page_num, map_page_num_content, 20)
                            
                            # Process the async generator and forward progress updates
                            final_chapters = None
                            async for update in ocr_generator:
                                # Check if this is the final result (not a string)
                                if not isinstance(update, str):
                                    final_chapters = update
                                else:
                                    # Forward progress updates
                                    yield update

                            if final_chapters:
                                # Step 8: Create and return metadata
                                yield f"data: OCR processing succeeded for book_id: {book_id}, total chapters: {len(final_chapters)}\n\n"
                                book_doc = create_book_metadata(book_id, (end_page_num-first_page_num), first_page_num, final_chapters, full_text, error_msg)
                                print("Book metadata: ", book_doc)

                                if not book_doc["success"]:
                                    print("Error in processing book chunks: ", book_doc.get("error", "Unknown error"))
                                    yield f"data: Error in chunking process: {json.dumps(book_doc)}\n\n"
                                else:
                                    number_of_chunks = book_doc["total_chunks"]
                                    yield f"data: Chunking completed. Total chunks: {number_of_chunks} for book_id: {book_id}\n\n"
                                    book_doc["book_id"] = book_id
                                    save_result = save_book_metadata_and_chunks(book_doc, final_chapters)
                                    
                                    if save_result["success"]:
                                        yield f"Book Metadata and Chunks Saved for {book_id}\n\n"
                                    else:
                                        yield f"Error saving book data for {book_id}: {save_result['error']}\n\n"
                            else:
                                yield f"data: OCR processing failed for book_id: {book_id}. {error_msg}\n\n"
                        else:
                            book_doc = create_book_metadata(
                                book_id, (end_page_num-first_page_num), first_page_num,
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


def save_book_metadata_and_chunks(book_metadata: dict, chapter_data: dict):
    """Save both book metadata and chunks in a single transaction."""
    try:
        # Start a session for the transaction
        with client.start_session() as session:
            with session.start_transaction():
                # Delete existing metadata and chunks for this book_id
                print(f"Book metadata = {book_metadata}")
                book_id = book_metadata["book_id"]
                
                # Remove existing metadata
                if book_chunk_metadata_collection.find_one({"file_id": book_id}, session=session):
                    book_chunk_metadata_collection.delete_one({"file_id": book_id}, session=session)

                # Remove existing chunks
                if book_chunk_collection.find_one({"book_id": book_id}, session=session):
                    book_chunk_collection.delete_one({"book_id": book_id}, session=session)

                print(f"Delete existing metadata and chunks for book_id: {book_id}")
                # Insert new metadata and chunks
                metadata_result = book_chunk_metadata_collection.insert_one(book_metadata, session=session)
                print(f"Book metadata inserted")
                chunks_result = book_chunk_collection.insert_many(chapter_data, session=session)
                print(f"Book chunks inserted")
                
                return {
                    "metadata_id": metadata_result.inserted_id,
                    "chunks_id": chunks_result.inserted_ids,
                    "success": True
                }
    except Exception as e:
        # If transaction fails (e.g., no replica set), fall back to individual operations
        print(f"Saving book metadata and/or chunks have failed:  {str(e)}")
        return {
            "metadata_id": -1,
            "chunks_id": -1,
            "success": False
        }


def save_book_metadata(book_metadata: dict):
    # Remove any prior metadata entries since only one should be present
    book_id_remove = book_metadata["book_id"]
    if book_chunk_metadata_collection.find_one({"file_id": book_id_remove}):
        book_chunk_metadata_collection.delete_one({"file_id": book_id_remove})
    return book_chunk_metadata_collection.insert_one(book_metadata).inserted_id


def save_book_chunks(chapter_data: dict):
    # Remove any prior metadata entries since only one should be present
    book_id_remove = chapter_data["book_id"]
    if book_chunk_collection.find_one({"book_id": book_id_remove}):
        book_chunk_collection.delete_one({"book_id": book_id_remove})
    return book_chunk_collection.insert_one(chapter_data).inserted_id


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


def process_chapters_from_text(full_text: str, book_id: str):
    """Split text into chapters and handle long chapters."""
    chapters = split_into_chapters(full_text, book_id)
    print(f"Total chapters extracted: {len(chapters)}")

    if chapters:
        return split_long_chapters(chapters, 8000, book_id)
    return []


def split_into_chapters(full_text, book_id):
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
            "book_id": book_id,
            "chapter": chapter_number,
            "text": chapter_text
        })

    return chapters


def split_long_chapters(chapters, max_chars: int, book_id: str):
    final_chunks = []
    page_number = 1

    for ch in chapters:
        text = ch["text"]

        if len(text) <= max_chars:
            final_chunks.append({
                "book_id": book_id,
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
                    "book_id": book_id,
                    "chapter": ch["chapter"],
                    "part": labels[idx],
                    "text": part_text,
                    "page_number": page_number
                })
                if idx < len(parts) - 1:
                    page_number += 1

        page_number += 1

    return final_chunks


async def process_with_ocr(pdf_path: str, book_id: str, first_page_num: int, map_page_num_content: dict, ocr_page_count):
    # Convert pages to images
    page_nums = list(map_page_num_content.keys())[:ocr_page_count]
    yield f"data: Converting pages to images for book_id: {book_id}, page count: {ocr_page_count}\n\n"
    await asyncio.sleep(0.5)
    convert_pages_to_images(pdf_path, book_id, page_nums, ocr_page_count)

    # Extract text from OCR
    # end_page_num = first_page_num + ocr_page_count
    # publish_book_events_for_OCR(book_id, first_page_num, end_page_num)
    
    items_processed = False
    
    # Process the async generator and forward progress updates
    async for update in check_ocr_processed_pages(book_id, page_nums, ocr_page_count):
        print(update)

        # Check if this is the final completion message
        if "All OCR pages processed" in update:
            items_processed = True
            break
    
    if items_processed:
        yield f"data: OCR processing completed for book_id: {book_id}. Scanning completed for {ocr_page_count} pages. Extracting chapter headings...\n\n"
    else:
        yield f"data: OCR processing stalled. Please try again later.\n\n"
    await asyncio.sleep(0.5)
    # Filter chapter headings
    # The logic in this method will differ for each book based on how the chapter no image is encoded
    map_page_num_page_heading = filter_chapter_headings_for_chapter_beginning(first_page_num, first_page_num + ocr_page_count, book_id)

    if not map_page_num_page_heading:
        print("Unable to extract text using OCR.")
        yield f"data: Unable to extract text using OCR.\n\n"
        await asyncio.sleep(0.5)

    # # Get non-blank chapter headings
    map_page_num_chapter_heading = {
         page_no: page_heading
         for page_no, page_heading in map_page_num_page_heading.items() if page_heading
    }

    final_chapters = {}
    if not map_page_num_chapter_heading:
        print("Chapter headings are blank. Unable to proceed.")
        yield f"data: Chapter headings are blank. Unable to proceed.\n\n"
        await asyncio.sleep(0.5)
    else:
        # Add chapter prefixes
        chapter_no = 1
        full_text = ""
        for page_no in map_page_num_chapter_heading:
            page_text = map_page_num_content[page_no]
            page_prefix = f"Chapter {chapter_no}\n\n"
            page_text = page_prefix + page_text
            chapter_no += 1
            page_text = clean_text(page_text)
            full_text += page_text + " "

        # Reprocess to split by chapters
        chapters = split_into_chapters(full_text, book_id)
        print(f"Total chapters extracted after OCR: {len(chapters)}")
        final_chapters = split_long_chapters(chapters, 8000, book_id)

    yield f"data: Perform Cleanup\n\n"
    await asyncio.sleep(0.5)
    # Clean up
    cleanup_ocr_files(book_id)
    if final_chapters:
        yield final_chapters
    else:
        yield None

file_path_prefix = "/home/koushick/sarah-maas-pages-for-OCR"


async def check_ocr_processed_pages(book_id: str, page_nums: [], ocr_page_count: int):
    processed_count = sm_map_page_nos_chap_heading_collection.count_documents(
        {"page_num": {"$in": page_nums}, "book_id": book_id})
    exponential_backoff_count = 0
    base_wait_time = 5
    while True:
        if processed_count >= ocr_page_count:
            print("All OCR pages processed.")
            yield f"data: All OCR pages processed.\n\n"
            await asyncio.sleep(0.5)
            break
        yield f"data: Scanned and analyzed {processed_count} pages so far...\n\n"
        print(f"Scanning pages to analyze...{processed_count} scanned so far.")

        # Calculate wait time with exponential backoff
        time_to_wait = base_wait_time * (2 ** exponential_backoff_count)
        print(f"Waiting for {time_to_wait} seconds (backoff level: {exponential_backoff_count})")

        seconds_index = 0
        while seconds_index < time_to_wait:
            yield f"data: Will provide next update in {time_to_wait - seconds_index} seconds\n\n"
            await asyncio.sleep(1)
            seconds_index += 1

        # Refresh the count
        prev_processed_count = processed_count
        processed_count = sm_map_page_nos_chap_heading_collection.count_documents(
            {"page_num": {"$in": page_nums}, "book_id": book_id})

        # Check for no progress condition
        if prev_processed_count != 0 and processed_count == prev_processed_count:
            exponential_backoff_count += 1
            if exponential_backoff_count >= 4:
                print("Maximum exponential backoff reached (4 doublings), breaking to avoid infinite loop.")
                yield f"data: Maximum exponential backoff reached (4 doublings), breaking to avoid infinite loop.\n\n"
                await asyncio.sleep(0.5)
                break
            print(f"No progress in OCR processing, applying exponential backoff (level {exponential_backoff_count}).")
        else:
            # Reset backoff count when progress is made
            exponential_backoff_count = 0


def cleanup_ocr_files(book_id: str):
    """Clean up temporary OCR files."""
    try:
        if os.path.exists(file_path_prefix):
            # Remove files containing book_id
            for file in os.listdir(file_path_prefix):
                if book_id in file:
                    os.remove(os.path.join(file_path_prefix, file))
        if os.path.exists("map_of_page_nos_chapter_heading.json"):
            os.remove("map_of_page_nos_chapter_heading.json")
        print(f"Cleaned up JPG files created for {book_id}")
    except Exception as e:
        print(f"Error during cleanup: {e}")


def convert_pages_to_images(pdf_path: str, book_id: str, page_nums: list, max_pages):
    """Convert PDF pages to images for OCR processing."""
    pdf_doc = fitz.open(pdf_path)
    os.makedirs(file_path_prefix, exist_ok=True)

    count = 0
    for page_num in page_nums:
        if count >= max_pages:
            break

        file_path = f"page_{book_id}_{page_num}.jpg"
        output_path = f"{file_path_prefix}/{file_path}"

        if not os.path.exists(output_path):
            # Convert to image
            fitz_page = pdf_doc[page_num - 1]
            pix = fitz_page.get_pixmap(dpi=300)
            pix.save(output_path)
            print(f"✓ Created: {output_path}")

        if sm_map_page_nos_chap_heading_collection.find_one({"page_num": page_num}):
            print(f"Page {page_num} found in database, skipping...")
        else:
            print(f"Sending message for page {page_num} with image path {file_path}")
            producer.send(topic, {"book_id": book_id, "page_num": page_num, "image_path": file_path})

        count += 1

    pdf_doc.close()
    return count


def publish_book_events_for_OCR(book_id: str, start_page, end_page_num: int):
    """Publish book events for OCR processing."""
    try:
        page_index = start_page
        print("Starting to send messages...")
        while page_index < end_page_num:
            if sm_map_page_nos_chap_heading_collection.find_one({"page_num": page_index}):
                print(f"Page {page_index} found in database, skipping...")
                page_index += 1
                continue  # Skip pages already in the database
            image_path = f"page_{book_id}_{page_index}.jpg"
            print(f"Sending message for page {page_index} with image path {image_path}")
            producer.send(topic, {"book_id": book_id, "page_num": page_index, "image_path": image_path})
            page_index += 1
        
    except Exception as e:
        print(f"Error: {str(e)}")