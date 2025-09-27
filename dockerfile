FROM python:3.11-slim

# Set working directory
WORKDIR /app

ENV VAULT_ADDR=64.227.147.196:8200
ENV UI_ORIGIN_URL=localhost:4200
ENV VAULT_RETRIEVER_URL=64.227.147.196:8300

# Copy requirements and install dependencies
COPY requirements.txt .
COPY encrypted*.txt ./
COPY vault-droplet/ssl/ca.crt ./vault-droplet/ssl/ca.crt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY Sarah_Maas_Chatbot_Crescent_City.py ./sarah_maas_chatbot_crescent_city.py

# Expose FastAPI default port
EXPOSE 8000

# Start FastAPI app
CMD ["sh", "-c", "uvicorn sarah_maas_chatbot_crescent_city:app --port 8000 > logfile.txt 2>&1"]
#CMD ["sh", "-c", "uvicorn sarah_maas_chatbot_crescent_city:app --host 0.0.0.0 --port 8000"]