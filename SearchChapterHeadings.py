from tinydb import TinyDB, Query
db = TinyDB('map_of_page_nos_chapter_heading.json')


ChapterHeading = Query()

def search_chapter_heading_in_db() -> list[str]:
    lines: list[str] = []
    for page_num in range(21, 550):
        record = db.get(ChapterHeading.page_num == page_num)
        if record is not None:
            lines.append(str(record['extracted_text']))
    return lines  # Append empty line as separator

def cleanup_text(text: str) -> str:
    # Remove unwanted characters and normalize spaces
    text = re.sub(r'[\r\n\t]+', '', text)  # Replace newlines and tabs with space
    text = text.strip()  # Trim leading/trailing spaces
    return text
    
import re

if __name__ == "__main__":
       
    # Your document text
    lines = search_chapter_heading_in_db()

    # Pattern: Match empty OR (non-ASCII chars + optional digits/symbols)
    # Exclude any line with ASCII letters
    filtered = []
    for line in lines:
        # if "\n" in line:
        #     line = line.replace("\n", "")
        line = cleanup_text(line)

        # Skip empty or whitespace-only lines
        if not line or not line.strip():
            line='(empty)'
            filtered.append(line)

        # Skip lines with alphabets
        if re.search(r'[a-zA-Z]', line):
            # But keep if it has digits
            if re.search(r'\d', line) or re.search(r'^[a-zA-Z]{1,3}$', line):
                filtered.append(line)
            continue

        # Keep if has non-ASCII or digits
        if re.search(r'[^\x00-\x7f]', line) or re.search(r'\d', line):
            filtered.append(line)

    for line in filtered:
        print(line)