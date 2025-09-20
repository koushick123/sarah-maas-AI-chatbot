FROM python:3.11-slim

# Set environment variable for Fernet key
ENV FERNET_KEY=
ENV FERNET_KEY_MONGO_USERNAME=
ENV FERNET_KEY_MONGO_PASSWORD=

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
COPY encryptedopenapi.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY Sarah-Maas-Chatbot-Crescent-City.py .

# Expose FastAPI default port
EXPOSE 8000

# Start FastAPI app
CMD ["sh", "-c", "uvicorn Sarah-Maas-Chatbot-Crescent-City:app --host 0.0.0.0 --port 8000 > logfile.txt 2>&1"]
#CMD ["sh", "-c", "uvicorn Sarah-Maas-Chatbot-Crescent-City:app --host 0.0.0.0 --port 8000"]