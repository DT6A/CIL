import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse


import numpy as np
from tqdm.auto import tqdm
import pandas as pd

# from vllm import LLM, SamplingParams

import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.optim import AdamW

from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import get_scheduler

import evaluate


class TweetsDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]["text"], self.data[idx]["label"]


def get_args():
    parser = argparse.ArgumentParser(description='Tuning')
    parser.add_argument(
        '--run_name', type=str, default="test", help='Wandb run name'
    )
    parser.add_argument(
        '--model_name', type=str, default="google-bert/bert-base-uncased", help='Model to train'
    )
    parser.add_argument(
        '--batch_size', type=int, default=32, help='Batch size'
    )
    parser.add_argument(
        '--checkpoint', type=str, default='checkpoint_best', help='Checkpoint to load'
    )
    parser.add_argument('--checkpoint_dir', type=str, default='.', help='Checkpoint directory')
    parser.add_argument('--use_trained_tokenizer', action=argparse.BooleanOptionalAction, help='Use a trained tokenizer')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    tweets = []
    labels = []


    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(line.lower().rstrip())
                labels.append(label)

    load_tweets('twitter-datasets/test_data.txt', 0)
    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    if not args.use_trained_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"tokenizers/{args.model_name}")

    def tokenize_function(examples):
        if "flan" not in args.run_name:
            return tokenizer(examples, padding="max_length", truncation=True)
        else:
            return tokenizer(examples, padding="max_length", return_token_type_ids=False)

    np.random.seed(1)  # Reproducibility!

    eval_dataset = TweetsDataset([
        {
            "text": text,
            "label": l
        }
        for text, l in zip(tweets, labels)
    ])

    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

    model = AutoModelForSequenceClassification.from_pretrained(
        f"/{args.checkpoint_dir}/checkpoints/{args.run_name}/{args.checkpoint}"
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    def prepare_batch(batch):
        texts, labels = batch
        batch = tokenize_function(texts)
        batch.update({
            "labels": torch.LongTensor(labels),
        })
        batch["attention_mask"] = torch.LongTensor(batch["attention_mask"])
        if "token_type_ids" in batch:
            batch["token_type_ids"] = torch.LongTensor(batch["token_type_ids"])
        batch["input_ids"] = torch.LongTensor(batch["input_ids"])
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch


    model.eval()
    result = []
    for batch in tqdm(eval_dataloader):
        batch = prepare_batch(batch)
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        result += predictions.cpu().tolist()
    id_column = list(range(1, 10001))
    result = [2 * r - 1 for r in result]
    df = pd.DataFrame({
        'Id': id_column,
        'Prediction': result
    })
    df.to_csv(f'{args.run_name}.csv', index=False)
