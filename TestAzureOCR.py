from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from Sarah_Maas_Chatbot_Crescent_City import decrypt_azure_ocr_api, decrypt_azure_ocr_host

# Your Azure credentials
endpoint = decrypt_azure_ocr_host()
subscription_key = decrypt_azure_ocr_api()

from PIL import Image
import io

def azure_ocr_extract_header(image_path: str) -> list[str]:
    """
    Crop the top region of the page to force OCR to detect page/section numbers.
    """
    try:
        # Load image with Pillow
        img = Image.open(image_path)
        w, h = img.size

        # Crop top 15% of page — adjust if needed
        header_height = int(h * 0.16)
        header_crop = img.crop((0, 0, w, header_height))

        # Convert cropped image to bytes
        buffer = io.BytesIO()
        header_crop.save(buffer, format="PNG")
        header_bytes = buffer.getvalue()

        # Run Azure OCR on header only
        client = ImageAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(subscription_key)
        )

        analysis = client.analyze(
            image_data=header_bytes,
            visual_features=[VisualFeatures.READ]
        )

        results = []
        if analysis.read:
            for block in analysis.read.blocks:
                for line in block.lines:
                    results.append(line.text)
                    for word in line.words:
                        results.append(word.text)

        return results

    except Exception as e:
        print("Header OCR error:", e)
        return []

from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from azure.cognitiveservices.vision.computervision.models import OperationStatusCodes
from msrest.authentication import CognitiveServicesCredentials
import time

credentials = CognitiveServicesCredentials(subscription_key)
client = ComputerVisionClient(endpoint, credentials)

def ocr_image(image_path):

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

    api_count = 0

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
        if api_count == 18:
            print("Wait for 1 minute to avoid 429 error from Azure OCR API...")
            time.sleep(60)
            print("Resuming...")
            api_count = 1
        print("Wait for 2 secs before checking OCR result...")
        time.sleep(5)
        api_count += 1

    # Extract text from result
    text = []
    if read_result.status == OperationStatusCodes.succeeded:
        for page in read_result.analyze_result.read_results:
            for line in page.lines:
                text.append(line.text)

    return "\n".join(text)


if __name__ == "__main__":
    #test_image_path = "output_default.jpg"  # Replace with your test image path
    test_image_path = "pages_for_OCR/page_empire-of-storms-20251105130926-b633ff79_22.jpg"  # Replace with your test image path
    #text = azure_ocr_extract_text(test_image_path)
    text =ocr_image(test_image_path)
    print("Extracted Text:")
    print(text)