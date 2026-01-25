import re
from pymongo import MongoClient
import urllib.parse
from SarahMaasChatbotCrescentCity import decrypt_mongo_user, decrypt_mongo_password, decrypt_mongo_hosturl

username = urllib.parse.quote_plus(decrypt_mongo_user())
password = urllib.parse.quote_plus(decrypt_mongo_password())
host_url = decrypt_mongo_hosturl()
uri = f"mongodb+srv://{username}:{password}@{host_url}/?retryWrites=true&w=majority&appName=dev-cluster"

client = MongoClient(uri)
chapter_heading_collection = client["sarah-maas-db"]['sarah-maas-map-page-nos-chapter-heading']

def extract_chapter_text_from_db(page_num,book_id) -> str:
    record = chapter_heading_collection.find_one({"page_num": page_num, "book_id": book_id})
    if record is not None:
        return str(record['extracted_text'])
    return ""

def cleanup_text(text: str) -> str:
    # Remove unwanted characters and normalize spaces
    text = re.sub(r'[\r\n\t]+', '', text)  # Replace newlines and tabs with space
    text = text.strip()  # Trim leading/trailing spaces
    return text
    

def filter_chapter_headings_for_chapter_beginning(start, end, book_id) -> dict[int, str]:
    page_num_with_chapter_headings = {}
    chapter_page_range = {}
    chapter_no = 1
    page_range = []
    while start < end:
        line = extract_chapter_text_from_db(start, book_id)
        # Pattern: Match empty OR (non-ASCII chars + optional digits/symbols)
        # Exclude any line with ASCII letters
        if check_if_chapter_heading(line):
            page_num_with_chapter_headings[start] = line
            print(f"Chapter heading found on page {start}: {line}")
            chapter_page_range[chapter_no-1] = page_range
            print(f"Chapter page range = {chapter_page_range}")
            chapter_no += 1
            page_range.clear()
        page_range.append(start)
        start += 1
    return page_num_with_chapter_headings

def check_if_chapter_heading(line: str) -> bool:
    line = cleanup_text(line)

    # Skip empty or whitespace-only lines
    if not line or not line.strip():
        line='(empty)'
        return True
    
    # Skip lines with alphabets
    if re.search(r'[a-zA-Z]', line):
        # But keep if it has digits or has anywhere between 1 to 3 letters only
        if re.search(r'\d', line) or re.search(r'^[a-zA-Z]{1,3}$', line):
            return True
        
    # Keep if has non-ASCII or digits
    if re.search(r'[^\x00-\x7f]', line) or re.search(r'\d', line):
        return True
    
    return False