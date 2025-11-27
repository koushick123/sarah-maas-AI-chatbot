from azure.ai.vision.imageanalysis import ImageAnalysisClient
from azure.ai.vision.imageanalysis.models import VisualFeatures
from azure.core.credentials import AzureKeyCredential
from Sarah_Maas_Chatbot_Crescent_City import decrypt_azure_ocr_api, decrypt_azure_ocr_host

# Your Azure credentials
endpoint = decrypt_azure_ocr_host()
subscription_key = decrypt_azure_ocr_api()

def azure_ocr_extract_text(image_path: str) -> str: 
    try:
        client = ImageAnalysisClient(
            endpoint=endpoint,
            credential=AzureKeyCredential(subscription_key)
        )

        with open(image_path, "rb") as image_stream:
            analysis = client.analyze(
                image_data=image_stream.read(),
                visual_features=[VisualFeatures.READ]
            )

        extracted_text = []
        if analysis.read:
            for line in analysis.read.blocks:
                extracted_text.append(line.lines)
                extracted_text.append("\n")

        return extracted_text
    except Exception as e:
        print(f"Error during Azure OCR: {str(e)}")
        return ""
    
if __name__ == "__main__":
    test_image_path = "output_default.jpg"  # Replace with your test image path
    text = azure_ocr_extract_text(test_image_path)
    print("Extracted Text:")
    print(text)