import torch
from llm_model import LargeLanguageModel
from fastapi import FastAPI, HTTPException
import logging

app = FastAPI()
logging.basicConfig(level = logging.INFO)

# Load model and vocabulary
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size, d_model, num_heads, num_layers, d_ff, seq_len = 10000, 256, 8, 4, 512, 32
model = LargeLanguageModel(vocab_size, d_model, num_layers, d_ff, seq_len).to(device)
model.load_state_dict(torch.load('model.pt'))
model.eval()
word2idx = torch.load('word2idx.pt')
idx2word = torch.load('idx2word.pt')

@app.post('/generate')
async def generate(data: dict):
    try:
        prompt = data.get('prompt', '').lower()
        max_length = data.get('max_length', 50)

        # Tokenize input
        words = prompt.split()
        tokens = [word2idx.get(word, word2idx['<UNK>']) for word in words][-seq_len:]
        input_ids = torch.tensor([tokens], dtype = torch.long).to(device)

        # Generate text
        generate = []
        with torch.no_grad():
            for _ in range(max_length):
                mask = torch.tril(torch.ones(len(input_ids[0]), len(input_ids[0]))).unsqueeze(0).unsqueeze(0).to(device)
                outputs = model(input_ids, mask)
                next_token = torch.argmax(outputs[:, -1, :], dim = -1)
                generate.append(next_token.items())
                input_ids = torch.cat([input_ids, next_token.unsqueeze(0).unsqueeze(0)], dim = -1)
                if len(input_ids[0]) > seq_len:
                    input_ids = input_ids[:, -seq_len:]

        # Decode output
        generate_text = ' '.join([idx2word.get(idx, '<UNK>') for idx in generate])
        logging.info(f"Generated text: {generate_text}")
        return {'generated_text': generate_text}
    except Exception as e:
        logging.error(f"Generation error: {str(e)}")
        raise HTTPException(status_code = 400, detail = str(e))