import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse


import numpy as np
from tqdm.auto import tqdm
import wandb

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
        '--batch_size', type=int, default=32, help='Train batch size'
    )
    parser.add_argument(
        '--eval_every', type=int, default=1000, help='Accuracy measure every N steps'
    )
    parser.add_argument(
        '--lr', type=float, default=5e-5, help='Learning rate'
    )
    parser.add_argument(
        '--n_epochs', type=int, default=1, help='Number of epochs'
    )
    parser.add_argument(
        '--train_samples', type=int, default=None, help='Size of the training subset (None for full)'
    )
    parser.add_argument(
        '--subset',
        action='store_true',
        default=False,
        help='Use small training set'
    )
    parser.add_argument(
        '--disable_wandb',
        action='store_true',
        default=False,
        help='Stop using wandb'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = get_args()
    if not args.disable_wandb:
        wandb.init(
            project="CIL-logs",
            config=vars(args),
            name=args.run_name,
            save_code=True,
        )

    tweets = []
    labels = []


    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(line.lower().rstrip())
                labels.append(label)

    if not args.subset:
        load_tweets('twitter-datasets/train_neg_full.txt', 0)
        load_tweets('twitter-datasets/train_pos_full.txt', 1)
    else:
        load_tweets('twitter-datasets/train_neg.txt', 0)
        load_tweets('twitter-datasets/train_pos.txt', 1)

    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    filtered_tweets = []
    filtered_labels = []
    tweets_dict = {}

    for t, l in zip(tweets, labels):
        if t not in tweets_dict:
            filtered_tweets.append(t)
            filtered_labels.append(l)
        tweets_dict[t] = 1

    tweets = np.array(filtered_tweets)
    labels = np.array(filtered_labels)

    hash_neg = {}
    hash_pos = {}

    # for t, l in tqdm.tqdm(list(zip(tweets, labels))):
    #     tokens = t.split()
    #     for token in tokens:
    #         if "#" in token:
    #             if l == 0:
    #                 if token not in hash_neg:
    #                     hash_neg[token] = 0
    #                 hash_neg[token] += 1
    #             else:
    #                 if token not in hash_pos:
    #                     hash_pos[token] = 0
    #                 hash_pos[token] += 1
    #
    # most_frequent_neg = sorted(hash_neg, key=hash_neg.get, reverse=True)[:20]
    # most_frequent_pos = sorted(hash_pos, key=hash_pos.get, reverse=True)[:20]
    #
    # print("=" * 30)
    # print("Most negative hashtags")
    # print("=" * 30)
    # for token in most_frequent_neg:
    #     if token == "#":
    #         continue
    #     print(token, hash_neg[token], f"{hash_neg[token] / len(tweets) * 100}%")
    #
    # print("=" * 30)
    # print("Most positive hashtags")
    # print("=" * 30)
    # for token in most_frequent_pos:
    #     if token == "#":
    #         continue
    #     print(token, hash_pos[token], f"{hash_pos[token] / len(tweets) * 100}%")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    def tokenize_function(examples):
        return tokenizer(examples, padding="max_length", truncation=True)

    np.random.seed(1)  # Reproducibility!

    shuffled_indices = np.random.permutation(len(tweets))
    split_idx = int(0.999 * len(tweets))

    train_indices = shuffled_indices[:split_idx]
    val_indices = shuffled_indices[split_idx:]

    if args.train_samples is not None:
        train_indices = train_indices[:args.train_samples]
        val_indices = val_indices[:1000]

    print("Train/Val sizes:", len(train_indices), len(val_indices))
    train_dataset = TweetsDataset([
        {
            "text": text,
            "label": l
        }
        for text, l in zip(tweets[train_indices], labels[train_indices])
    ])
    val_dataset = TweetsDataset([
        {
            "text": text,
            "label": l
        }
        for text, l in zip(tweets[val_indices], labels[val_indices])
    ])

    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    eval_dataloader = DataLoader(val_dataset, batch_size=args.batch_size * 2)

    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    num_epochs = args.n_epochs
    num_training_steps = num_epochs * len(train_dataloader)
    lr_scheduler = get_scheduler(
        name="linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model.to(device)

    progress_bar = tqdm(range(num_training_steps))

    model.train()

    def prepare_batch(batch):
        texts, labels = batch
        batch = tokenize_function(texts)
        batch.update({
            "labels": torch.LongTensor(labels),
        })
        batch["attention_mask"] = torch.LongTensor(batch["attention_mask"])
        batch["token_type_ids"] = torch.LongTensor(batch["token_type_ids"])
        batch["input_ids"] = torch.LongTensor(batch["input_ids"])
        batch = {k: v.to(device) for k, v in batch.items()}
        return batch

    global_step = 0
    best_accuracy = 0
    for epoch in range(num_epochs):
        model.train()
        for batch in train_dataloader:
            batch = prepare_batch(batch)

            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            global_step += 1

            if not args.disable_wandb:
                wandb.log(
                    {"loss": loss.item()}, step=global_step
                )

            if global_step % args.eval_every == 0:
                metric = evaluate.load("accuracy")
                model.eval()
                for batch in eval_dataloader:
                    batch = prepare_batch(batch)
                    with torch.no_grad():
                        outputs = model(**batch)

                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=-1)
                    metric.add_batch(predictions=predictions, references=batch["labels"])
                accuracy = metric.compute()
                print(accuracy)
                if best_accuracy < accuracy['accuracy']:
                    best_accuracy = accuracy['accuracy']
                    model.save_pretrained(
                        os.path.join(f"checkpoints/{args.run_name}", f"checkpoint_best"),
                        max_shard_size="500MB",
                    )
                if not args.disable_wandb:
                    wandb.log(
                        {"validation_accuracy": accuracy['accuracy']}, step=global_step
                    )
                model.train()

        metric = evaluate.load("accuracy")
        model.eval()
        for batch in eval_dataloader:
            batch = prepare_batch(batch)
            with torch.no_grad():
                outputs = model(**batch)

            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
        accuracy = metric.compute()
        if best_accuracy < accuracy['accuracy']:
            best_accuracy = accuracy['accuracy']
            model.save_pretrained(
                os.path.join(f"checkpoints/{args.run_name}", f"checkpoint_best"),
                max_shard_size="500MB",
            )
        print("Final accuracy:", accuracy)
        if not args.disable_wandb:
            wandb.log(
                {"validation_accuracy": accuracy['accuracy']}, step=global_step
            )
