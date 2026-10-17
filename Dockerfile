FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY workshop_a2a ./workshop_a2a
COPY ask.py .
