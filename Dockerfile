FROM python:3.9-slim

WORKDIR /templates

COPY requirements.txt .

RUN apt-get update && apt-get install -y gcc \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove gcc \
    && rm -rf /var/lib/apt/lists/*

COPY . .

EXPOSE 8080

CMD ["python", "run_pipeline.py"]