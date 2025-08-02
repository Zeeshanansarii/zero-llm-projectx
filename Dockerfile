FROM python:3.9-slim
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install DVC
RUN pip install dvc

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and model files
COPY . .
COPY model.pt model.pt
COPY model_finetuned_lora.pt model_finetuned_lora.pt
COPY word2idx.pt word2idx.pt
COPY word2idx_finetuned.pt word2idx_finetuned.pt
COPY idx2word.pt idx2word.pt
COPY idx2word_finetuned.pt idx2word_finetuned.pt
COPY documents.txt documents.txt

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV MILVUS_HOST=milvus-service
ENV MILVUS_PORT=19530
ENV SECRET_KEY=your-secure-secret-key

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "rag_app:app", "--host", "0.0.0.0", "--port", "8000"]