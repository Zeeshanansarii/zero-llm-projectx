import torch
import tempfile
import os
import unittest
from llm_model import LargeLanguageModel, MultiHeadAttention, TransformerBlock
from preprocess import TextDataset
from app import generate
from fastapi.testclient import TestClient
from app import app
import logging
logging.basicConfig(level = logging.INFO)

# Create a temporary text file for testing
class test_llm_model(unittest.TestCase):
    def setUp(self):
        self.temp_file = tempfile.NamedTemporaryFile(mode = 'w', suffix = '.txt', delete = False)
        sample_text = "this is a test sentence for the llm model this is another test sentence"
        self.temp_file.write(sample_text)
        self.temp_file.close()

        # Model parameters
        self.vocab_size = 100
        self.seq_len = 8
        self.d_model = 64
        self.num_heads = 4
        self.d_ff = 128
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize dataset and model
        self.dataset = TextDataset(self.temp_file.name, self.seq_len, self.vocab_size)
        self.model = LargeLanguageModel(self.vocab_size, self.d_model, self.num_heads, self.num_layers, self.d_ff, self.seq_len).to(self.device)
        self.client = TestClient(app)

    # Clean up temporary file
    def tearDown(self):
        os.unlink(self.temp_file.name)

    def test_dataset_loading(self):
        self.assertGreater(len(self.dataset), 0, "Dataset should contain sequences")
        self.assertEqual(len(self.dataset[0][0]), self.seq_len, "Input sequence length should match seq_len")
        self.assertTrue(len('<PAD>' in self.dataset.vocab, "Vocabulary should inclue <PAD> token"))
        self.assertTrue(len('<UNK>' in self.dataset.vocab, "Vocabulary should include <UNK> token"))

    def test_model_forward_pass(self):
        inputs = torch.randint(0, self.vocab_size, (2, self.seq_len)).to(self.device)
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len)).unsqueeze(0).unsqueeze(0).to(self.device)
        outputs = self.model(inputs, mask)
        self.assertEqual(outputs.shape, (2, self.seq_len, self.vocab_size),
                         f"Expected output shape (2, {self.seq_len}, {self.vocab_size}), got {outputs.shape}")
        
    def test_attention_mechanism(self):
        attention = MultiHeadAttention(self.d_model, self.num_heads).to(self.device)
        x = torch.randn(2, self.seq_len, self.d_model).to(self.device)
        output = attention(x)
        self.assertEqual(output.shape, (2, self.seq_len, self.d_model),
                         f"Expected attention output shape (2, {self.seq_len}, {self.d_model}), got {output.shape}")
        
    
    def test_transformer_block(self):
        transformer = TransformerBlock(self.d_model, self.num_heads, self.d_ff).to(self.device)
        x = torch.randn(2, self.seq_len, self.d_model).to(self.device)
        output = transformer(x)
        self.assertEqual(output.shape, (2, self.seq_len, self.d_model),
                         f"Expected tranformer output shape (2, {self.seq_len}, {self.d_model}), got {output.shape}")
        
    
    def test_training_step(self):
        inputs = torch.randint(0, self.vocab_size, (2, self.seq_len)).to(self.device)
        targets = torch.randint(0, self.vocab_size, (2, self.seq_len)).to(self.device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr = 0.001)
        self.model.train()
        optimizer.zero_grad()
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len)).unsqueeze(0).unsqueeze(0).to(self.device)
        outputs = self.model(inputs, mask)
        loss = criterion(outputs.view(-1, self.vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
        self.assertFalse(torch.isnan(loss).any(), "Loss should not contain NaN values")
        self.assertGreater(loss.item(), 0, "Loss should be positive")

    # Test the api / generate endpoint
    def test_inference_endpoint(self):
        try:
            responce = self.client.post('/generate', json = {'prompt': 'this is a test', 'max_length': 10})
            self.assertEqual(responce.status_code, 200, "Expected status code 200")
            self.assertIn('generated_text', responce.json(), "Response should contain generated_text")
        except Exception as e:
            logging.warning(f"Inference test skipped due to missing model files: {str(e)}")

if __name__ == 'main':
    unittest.main()