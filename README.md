The sarah-maas-AI-chatbot application requires sarah-maas-vault repo application to be up and running.
The sarah-maas-vault is a custom HashiCorp vault setup with credentials needed to connect to LLMs and MongoDB.

Pre-requisite
Run sarah-maas-vault as a docker. Refer sarah-maas-vault repo for additional details.

NOTE:
To run inside codespaces, use **run-sm-app.sh** script, instead of running from VS Code directly.
The script runs uvicorn Fast API directly without issues.

Run sudo apt-get install tesseract-ocr for OCR support.