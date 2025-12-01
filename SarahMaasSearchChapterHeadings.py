import re
from tinydb import TinyDB, Query
db = TinyDB('map_of_page_nos_chapter_heading.json')

ChapterHeading = Query()

def extract_chapter_text_from_db(page_num) -> str:
    record = db.get(ChapterHeading.page_num == page_num)
    if record is not None:
        return str(record['extracted_text'])
    return ""

def cleanup_text(text: str) -> str:
    # Remove unwanted characters and normalize spaces
    text = re.sub(r'[\r\n\t]+', '', text)  # Replace newlines and tabs with space
    text = text.strip()  # Trim leading/trailing spaces
    return text
    

def filter_chapter_headings_for_chapter_beginning(start, end) -> dict[int, str]:
    page_num_with_chapter_headings = {}
    while start < end:
        line = extract_chapter_text_from_db(start)
        # Pattern: Match empty OR (non-ASCII chars + optional digits/symbols)
        # Exclude any line with ASCII letters
        if check_if_chapter_heading(line):
            page_num_with_chapter_headings[start] = line
            print(f"Chapter heading found on page {start}: {line}")
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