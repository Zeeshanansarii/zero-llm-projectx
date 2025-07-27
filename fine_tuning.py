import torch
import torch.nn as nn
import deepspeed
import os
import argparse
import numpy as np
import subprocess
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from llm_model import LargeLanguageModel
from preprocess import TextDataset
from prometheus_client import Counter, Histogram
from peft import LoraConfig, get_peft_model
from ray import tune
from ray.tune.schedulers import ASHAScheduler
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from torch.optim.lr_scheduler import LambdaLR
import logging

# Configure logging
logging.basicConfig(level = logging.INFO,
                    format = '%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Prometheus metrics 
fine_tune_loss = Histogram('fine_tune_loss', 'Fine-tuning loss per epoch')
fine_tune_steps = Counter('fine_tune_steps_total', 'Total fine-tuning steps')
drift_detected = Counter('fine_drift_detected_total', 'Total data drift detections')

def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / warmup_steps
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))
    return LambdaLR(optimizer, lr_lambda)

def fine_tune_model(config, data_file = 'data.txt', checkpoint_dir = None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Fine-tuning on device: {device}")

    # Load dataset
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Data file {data_file} not found")
    dataset = TextDataset(data_file, config['seq-len'], config['vocab_size'])
    DataLoader = DataLoader(dataset, batch_size = config['batch_size'], shuffle = True)

    # Initialize model with LoRA
    model = LargeLanguageModel(config['vocab_size'], config['d_model'], config['num_heads'],
                               config['num_layers'], config['d_ff'], config['seq-len']).to(device)
    try:
        model.load_state_dict(torch.load('model.pt', map_location = device))
        logger.info("Loaded pre-trained model weights")
    except FileNotFoundError:
        logger.warning("Pre-trained model not found, starting from scratch")

    Lora_config = LoraConfig(r = 8, lora_aplha = 16, target_modules = ["W_q", "W_k", "W_v", "W_o"], lora_droput = 0.1)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Deepspeed configuration
    ds_config = {
        "train_batch_size": config['batch_size'] * config['accum_steps'],
        "fp16": {"enabled": True},
        "optimizer": {
            "type": "AdamW",
            "params": {"lr": config['learning_rate'], "weight_decay": 0.01}
        },
        "gradient_accumulation_steps": config['accum_steps'],
        "gradient_clipping": 1.0
    }

    model_engine, optimizer, _, _ = deepspeed.initialize(model = model, config = ds_config)
    scaler = GradScaler()

    # Learning rate scheduler
    total_steps = len(dataloader) * config['epochs']
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps = 100, total_steps = total_steps)

    # Data drift detection
    ref_data = {'text': [dataset.text[:100]]}
    current_data = {'text': [dataset.text[:100]]}
    report = Report(metrics = [DataDriftPreset()])
    report.run(reference_data = ref_data, current_data= current_data)
    if report.as_dict()['metrics'][0]['result']['dataset_drift']:
        drift_detected.inc()
        logger.warning("Data drift detected in fine-tuning dataset")

    # Fine-tuning loop
    criterion = nn.CrossEntropyLoss()
    model_engine.train()
    best_loss = float('inf')

    for epoch in range(config['epochs']):
        total_loss = 0
        for batch_idx, (input, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            mask = torch.tril(torch.ones(config['seq-len'], config['seq-len'])).unsqueeze(0).unsqueeze(0).to(device)

            with autocast():
                outputs = model_engine(inputs, mask)
                loss = criterion(outputs.view(-1, config['vocab_size']), targets.view(-1))
                loss = loss / config['accum_steps']

            scaler.scale(loss).backward()
            fine_tune_steps.inc()

            if (batch_idx + 1) % config['accum_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * config['accum_steps']
            if batch_idx % 10 == 0:
                logger.info(f"Epoch {epoch+1} / {config['epochs']}, Batch{batch_idx}, Loss: {loss.item() * config['accum_steps']:.4f}")

        avg_loss = total_loss / len(dataloader)
        fine_tune_loss.observe(avg_loss)
        logger.info(f"Epoch {epoch + 1} / {config['epochs']} , Average Loss: {avg_loss:.4f}")
        scheduler.step()

        # Save checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            model_engine.save_checkpoint('model_finetuned_lora.pt')
            logger.info("Saved best fine-tuned model checkpoint")

        # Report to Ray Tune
        tune.report(mean_loss = avg_loss)

    # Save vocabulary and DVC versioning
    torch.save(dataset.word2idx, 'word2idx_finetuned.pt')
    torch.save(dataset.idx2word, 'idx2word_finetuned.pt')
    subprocess.run(['dvc', 'add', 'model_finetuned_lora.pt'])
    subprocess.run(['dvc', 'add', 'word2idx_finetuned.pt', 'idx2word_finetuned.pt'])
    subprocess.run(['dvc', 'push'])
    logger.info("Fine-tuning completed, model and vocabulary saved and DVC versioning")


def objective(config):
    fine_tune_model(config, data_file = 'data.txt')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = "Advanced fine-tuning of Large Language Model with LoRA and Deepspeed ")
    parser.add_argument('--data_file', type = str, default = 'data.txt', help = 'Path to fine-tuning dataset')
    args = parser.parse_args()

    # Ray tune hyperparameter search space
    config = {
        'vocab_size': 1000,
        'seq-len': 16,
        'd_model': 256,
        'num_heads': 8,
        'num_layers': 4,
        'd_ff': 512,
        'epochs': 10,
        'batch_size': tune.choice([16, 32, 64]),
        'accum_steps': tune.randint(2, 8),
        'learning_rate': tune.loguniform(1e-5, 1e-3)
    }

    scheduler = ASHAScheduler(metric = "mean_loss", mode = "min")
    tuner = tune.Tuner(
        objective,
        tune_config = tune.TuneConfig(
            scheduler = scheduler,
            num_samples = 10
        ),
        param_space = config
    )

    results = tuner.fit()
    best_config = results.get_best_config(metric = "mean_loss", mode = "min")
    logger.info(f"Best hyperparameters: {best_config}")