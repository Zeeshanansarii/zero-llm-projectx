import re
import torch
from collections import Counter
from torch.utils.data import Dataset
import logging
logging.basicConfig(level = logging.INFO)

class TextDataset(Dataset):
    def __init__(self, text_file, seq_len, vocab_size = 1000):
        self.seq_len = seq_len
        self.text = self.load_text(text_file)
        self.vocab, self.word2idx, self.idx2word = self.build_vocab(self.text, vocab_size)
        self.data = self.tokenize_text(self.text)

    def load_text(self, text_file):
        with open(text_file, 'r', encoding = 'utf-8') as f:
            text = f.read().lower()
            text = re.sub(r'[^a-z0-9\s]', '',text)
            return text
        
    def build_vocab(self, text, vocab_size):
        words = text.split()
        word_counts = Counter(words)
        vocab = ['<PAD>', '<UNK>'] + [word for word, _ in word_counts.most_common(vocab_size - 2)]
        word2idx = {word : idx for idx, word in enumerate(vocab)}
        idx2word = {idx: word for word, idx in word2idx.items()}
        return vocab, word2idx, idx2word
    
    def tokenize_text(self, text):
        words = text.split()
        tokens = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in words]
        sequences = []
        for i in range(0, len(tokens) - self.seq_len, self.seq_len):
            sequences.append(tokens[i:i + self.seq_len + 1])
        return torch.tensor(sequences, dtype = torch.long)
    
    def __len_(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sequence = self.data[idx]
        return sequence[:-1], sequence[1:]