from PIL import Image
import io
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time
from Sarah_Maas_Chatbot_Crescent_City import decrypt_azure_ocr_api, decrypt_azure_ocr_host

# Your Azure credentials
endpoint = decrypt_azure_ocr_host()
subscription_key = decrypt_azure_ocr_api()

credentials = CognitiveServicesCredentials(subscription_key)
client = ComputerVisionClient(endpoint, credentials)

from tinydb import TinyDB, Query
db = TinyDB('map_of_page_nos_chapter_heading.json')

def read_text_from_cropped_ocr_image(image_path) -> str:

    # Load image with Pillow
    img = Image.open(image_path)
    w, h = img.size

    # Crop top 15% of page — adjust if needed
    header_height_start = int(h * 0.10)
    header_height_end = int(h * 0.15)
    header_crop = img.crop((0, header_height_start, w, header_height_end))

    # Convert cropped image to bytes
    buffer = io.BytesIO()
    header_crop.save(buffer, format="PNG")
    header_bytes = buffer.getvalue()   

    # Call API with image
    read_response = client.read_in_stream(io.BytesIO(header_bytes), raw=True)
    # Call API with image
    # read_response = client.read_in_stream(image_file, raw=True)

    # Get operation ID from response headers
    operation_id = read_response.headers["Operation-Location"].split("/")[-1]

    # Wait for the operation to complete
    while True:
        read_result = client.get_read_result(operation_id)
        if read_result.status not in [OperationStatusCodes.running, OperationStatusCodes.not_started]:
            break
        print(f"Wait for 2 secs before checking OCR result...")
        time.sleep(2)

    # Extract text from result
    text = []
    if read_result.status == OperationStatusCodes.succeeded:
        for page in read_result.analyze_result.read_results:
            for line in page.lines:
                text.append(line.text)

    return "\n".join(text)


def extract_and_save_text_from_ocr_page(start, end, book_id: str):
    ChapterMetaData = Query()
    page_index = start
    while page_index < end:
        if db.search(ChapterMetaData.page_num == page_index):
            print(f"Page {page_index} found in database, skipping...")
            page_index += 1
            continue  # Skip pages already in the database
        test_image_path = f"pages_for_OCR/page_{book_id}_{page_index}.jpg"
        try:
            text = read_text_from_cropped_ocr_image(test_image_path)
        except Exception as e:
            print(f"Error processing page {page_index} after wait: {str(e)}")
            if str(e).find("Too Many Requests") != -1:
                print("API call limit reached for free tier. Wait for a minute...")
                time.sleep(60)  # Wait for a minute before continuing
                print(f"Redo OCR processing for {page_index}")
                continue
        
        db.insert({'page_num': page_index, 'extracted_text': text[:20]})
        page_index += 1