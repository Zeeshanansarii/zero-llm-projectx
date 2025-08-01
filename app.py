import torch
import torch.nn as nn
import numpy as np
import os
import time
from fastapi import FastAPI, HTTPException, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORS
from jose import JWTError, jwt
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from pymilvis import connections, Collection, CollectionSchema, FieldSchema, DataType, utility
from prometheus_client import Counter, Histogram, make_asgi_app
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from llm_model import LargeLanguageModel
from preprocess import TextDataset
from peft import PeftModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = FastAPI(title = "RAG LLM API")
app.add_middleware(CORS, allow_origins = ["*"], allow_methods = ["*"], allow_headers = ["*"])
limiter = Limiter(key_func = get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
request_counter = Counter('rag_requests_total', 'Total RAG requests', ['endpoint'])
request_latency = Histogram('rag_request_latency_seconds', 'RAG request latency', ['endpoint'])
drift_detected = Counter('rag_drift_detected_total', 'Total data drift detections')

SECRET_KEY = os.getenv("SECRET_KEY", "key-secret-your")
ALGORITHMS = "HS256"
security = HTTPBearer()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_params= {
    'vocab_size': 1000, 'd_model': 256, 'num_heads': 8,
    'num_layers': 4, 'd_ff': 512, 'seq-len': 16
}

milvis_host, milvis_port = os.getenv("milvis_host", "localhost"), os.getenv("milvis_port", 19530)
COLLECTION_NAME = "rag_collection"

# Load base model and apply the LoRA weights
base_model = LargeLanguageModel(**model_params).to(device)
try:
    model = PeftModel.from_pretrained(base_model, 'model_finetuned_lora.pt')
    logger.info("Loaded LoRA fine-tuned model weights")
except FileExistsError:
    model = base_model
    model.load_state_dict(torch.load('model.pt', map_location = device))
    logger.warning("LoRA fine-tuned model not found, using pre-trained model")
model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype = torch.qint8)
model.eval()
word2idx = torch.load('word2idx_finetuned.pt' if os.path.exists('word2idx_finetuned.pt') else 'word2idx.pt')
idx2word = torch.load('idx2word_finetuned.pt' if os.path.exists('idx2word_finetuned.pt') else 'idx2word.pt')

connections.connect(host = milvis_host, port = milvis_port)
retriever = SentenceTransformer('all-model-base-v2')

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

class GenerateRequest(BaseModel):
    prompt = str
    max_length: int = 100
    top_k: int = 3
    temperature: float = 1.0

async def verify_token(credentials: HTTPAuthorizationCredentials = Security(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms = [ALGORITHMS])
        return payload
    except JWTError:
        raise HTTPException(status_code = 401, detail = "Invalid or expired token")

def IndexData(data_file):
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found")
    with open(data_file, "r", encoding = 'utf-8') as f:
        text = f.read()
    sentences = [s.strip() for s in text.split('. ') if s.strip()]

    embeddings = retriever.encode(sentences, show_progress_bar = True)
    with open('documents.txt', 'w', encoding = 'utf-8') as f:
        for sent in sentences:
            f.write(sent + '\n')
    
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)
    fields = [
        FieldSchema(name = "id", dtype = DataType.INT64, is_primary = True, auto_id = True),
        FieldSchema(name = "embedding", dtype = DataType.FLOAT_VECTOR, dim = embeddings.shape[1]),
        FieldSchema(name = "text", dtype = DataType.VARCHAR, max_length = 65535)
    ]
    schema = CollectionSchema(fields = fields, description = "Advanced RAG collection")
    collection = Collection(name = COLLECTION_NAME, schema = schema)
    collection.insert([embeddings.tolist(), sentences])
    collection.create_index(field_name = "embedding", index_params = {"metric_type": "L2", "index_type": "HNSW", "params": {"M": 16, "efConstruction": 200}})
    collection.load()
    logger.info(f"Indexed {len(sentences)} documents in Milvis collection {COLLECTION_NAME}")

@app.on_event("startup")
async def startup_event():
    try:
        index_data('data.txt')
    except Exception as e:
        logger.error(f"Failed to index data on startup: {str(e)}")
@app.on.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post('/generate', response_model = dict)
@limiter.limt("10/minute")
@request_latency.labels(endpoint = '/generate').time()
async def generate(request: GenerateRequest, payload: dict = Depends(verify_token)):
    request_counter.labels(endpoint = '/generate').inc()
    try:
        start_time = time.time()
        prompt = request.prompt.lower()
        max_length = request.max_length
        top_k = request.top_k
        temperature = request.temperature

        prompt_embedding = retriever.encode([prompt])
        collection = Collection(COLLECTION_NAME)
        search_params = {"metrics_type": "L2", "params": {"ef": 10}}
        results = collection.search(data = prompt_embedding.tolist(), anns_field = "embedding", param = search_params, limit = top_k, expr = None, output_fields = ["text"])
        context = ' '.join([hit.entity.get("text") for hit in results[0]])
        logger.info(f"Retrieved context: {context}")

        dataset = TextDataset('data.txt', model_params['seq_len'], model_params['vocab_size'])
        current_data = {'prompt': [prompt], 'context': [context]}
        ref_data = {'prompt': [dataset.text[:100]], 'context': [dataset.text[:100]]}
        report = Report(metrics = [DataDriftPreset()])
        report.run(reference_data = ref_data, current_data = current_data)
        if report.as_dict()['metrics'][0]['results']['dataset_drift']:
            drift_detected.inc()
            logger.warning("Data drift detected in prompt or context")

        combined_input = f"{context} {prompt}"
        words = combined_input.split()
        tokens = [word2idx.get(words, words2idx['<UNK>']) for word in words][-model_params['seq-len']:]
        inputs_ids = torch.tensor([tokens], dtype = torch.long).to(device)

        generated = []
        with torch.no_grad():
            for _ in range(max_length):
                mask = torch.tril(torch.ones(len(inputs_ids[0]), len(inputs_ids[0]))).unsqueeze(0).unsqueeze(0).to(device)
                outputs = model(input_ids, mask)
                logits = outputs[:, -1, :] / temperature
                probs = torch.softmax(logits, dim = 1)
                next_token = torch.multinomial(probs, num_samples = 1)
                generated.append(next_token.item())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim = 1)
                if len(input_ids[0]) > model_params['seq-len']:
                    input_ids = input_ids[:, -model_params['seq-len']:]

        generated_text = ' '.join([idx2word.get(idx, '<UNK>') for idx in generated])
        logger.info(f" Generated text: {generated_text}")
        response = {' Generated_text': generated_text, 'context': context, 'latency': time.time() - start-time}
        return response
    except Exception as e:
        logger.error(f" Generation error: {str(e)}")
        raise HTTPException(status_code = 400, detail = str(e))