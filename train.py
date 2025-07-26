import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from llm_model import LargeLanguageModel
from preprocess import TextDataset
import logging
logging.basicConfig(level=logging.INFO)

def train_model(data_file, vocab_size = 10000, seq_len = 32, d_model = 256, num_heads = 8,
                num_layers = 4, d_ff = 512, epochs = 15, batch_size = 64):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Training on device: {device}')

    # Load dataset
    dataset = TextDataset(data_file, seq_len, vocab_size)
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)

    # Initialize Model
    model = LargeLanguageModel(vocab_size, d_model, num_heads, num_layers,
                               d_ff, seq_len).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.paramters(), lr = 0.001)

    # Training the model
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Create causal mark
            mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0). unsqueeze(0).to(device)
            optimizer.zero_grad()
            outputs = model(inputs, mask)
            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            if batch_idx % 100 == 0:
                logging.info(f"Epoch {epoch+1} / {epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
        avg_loss = total_loss / len(dataloader)
        logging.info(f"Epoch {epoch+1} / {epochs}, Average Loss: {avg_loss:.4f}")

    # Save model
    torch.save(model.state_dict(), 'model.pt')
    torch.save(dataset.word2idx, 'word2idx')
    torch.save(dataset.idx2word, 'idx2word.pt')
    logging.info("Model and vocabulary saved")

"""if __name__ == "main":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type = str, required = True, help = '/home/zeeshan/project/zero-llm-projectx/data.txt')
    args = parser.parse_args()

    train_model(data_file = args.data_file) """

if __name__ == "main":
    train_model(data_file = "data.txt")