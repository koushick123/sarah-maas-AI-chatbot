from tinydb import TinyDB, Query
db = TinyDB('map_of_page_nos_chapter_heading.json')

import regex

ChapterHeading = Query()

def search_chapter_heading_in_db() -> list:
    lines = []
    for page_num in range(21, 550):
        record = db.get(ChapterHeading.page_num == page_num)
        if record is not None:
            lines.append(record['extracted_text'])
            lines.append('\n')
    return lines  # Append empty line as separator
            
    
import re

if __name__ == "__main__":
       
    # Your document text
    lines = search_chapter_heading_in_db()

    # Pattern: Match empty OR (non-ASCII chars + optional digits/symbols)
    # Exclude any line with ASCII letters
    pattern = re.compile(r'^(?:[^\x00-\x7fa-zA-Z]+|)$')

    filtered = [line for line in lines if not re.search(r'[a-zA-Z]', line) and 
                (not line or re.search(r'[^\x00-\x7f]', line))]

    # Alternative simpler version
    filtered = [
        line for line in lines 
        if not re.search(r'[a-zA-Z]', line)  # No ASCII letters
        and (not line.strip() or re.search(r'[^\x00-\x7f]', line))  # Empty OR has non-ASCII
    ]

    with open('filtered_results.txt', 'w', encoding='utf-8') as f:
        f.write(';'.join(filtered))

    print("--- Filtered Results ---")
    for idx, item in enumerate(filtered, 1):
        display = f"'{item}'" if item else "'' (empty)"
        if display != '\n':
            print(f"{idx}. {display.strip()}")