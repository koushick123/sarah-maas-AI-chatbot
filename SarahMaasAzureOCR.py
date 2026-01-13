from PIL import Image
import io
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time
from DecryptCredentials import decrypt_azure_ocr_api, decrypt_azure_ocr_host

# Your Azure credentials
endpoint = decrypt_azure_ocr_host()
subscription_key = decrypt_azure_ocr_api()

credentials = CognitiveServicesCredentials(subscription_key)
client = ComputerVisionClient(endpoint, credentials)

def read_text_from_cropped_ocr_image(image_path) -> str:

    start_image_ratio = 0.13
    end_image_ratio = 0.19
    extracted_text = []
    file_path_prefix = "pages_for_OCR/"
    # Iteratively increase the crop ratio until text is found or max ratio is reached
    while (end_image_ratio - start_image_ratio) <= 0.1 and not extracted_text:
        # Load image with Pillow
        img = Image.open(file_path_prefix + image_path)
        w, h = img.size

        header_height_start = int(h * start_image_ratio)
        header_height_end = int(h * end_image_ratio)
        header_crop = img.crop((0, header_height_start, w, header_height_end))

        # Convert cropped image to bytes
        buffer = io.BytesIO()
        header_crop.save(buffer, format="PNG")
        header_bytes = buffer.getvalue()

        try:
            # Call API with image
            read_response = client.read_in_stream(io.BytesIO(header_bytes), raw=True)

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

            if read_result.status == OperationStatusCodes.succeeded:
                for page in read_result.analyze_result.read_results:
                    for line in page.lines:
                        if line.text.strip():
                            extracted_text.append(line.text)
                print(f"Extracted text at ratio {start_image_ratio}-{end_image_ratio}: {extracted_text[:1]}")
        except Exception as e:
            print("Exception during OCR image processing: " + str(e))
            if str(e).find("Too Many Requests") != -1:
                print("Error during OCR image processing: " + str(e))
                print("API call limit reached for free tier. Wait for a minute...")
                time.sleep(60)  # Wait for a minute before continuing
                # Repeat the same ratio
                continue

        start_image_ratio -= 0.01
        
    return "".join(extracted_text)