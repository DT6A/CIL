import argparse
from typing import Optional
import numpy as np
from datasets import Dataset
import evaluate
import wandb
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from peft import LoraConfig, TaskType, get_peft_model

def trained_params_ratio(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad) / sum(
        p.numel() for p in model.parameters()
    )

def get_args():
    parser = argparse.ArgumentParser(description='Tuning')
    parser.add_argument('--run_name', type=str, default="test", help='Wandb run name')
    parser.add_argument('--model_name', type=str, default="google-bert/bert-base-uncased", help='Model to train')
    parser.add_argument('--batch_size', type=int, default=32, help='Train batch size')
    parser.add_argument('--eval_every', type=int, default=1000, help='Accuracy measure every N steps')
    parser.add_argument('--lr', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--n_epochs', type=int, default=1, help='Number of epochs')
    parser.add_argument('--train_samples', type=int, default=None, help='Size of the training subset (None for full)')
    parser.add_argument('--subset', action='store_true', default=False, help='Use small training set')
    parser.add_argument('--disable_wandb', action='store_true', default=False, help='Stop using wandb')
    parser.add_argument('--lora', action='store_true', default=False, help='Use LoRA')
    parser.add_argument('--lora_rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora_alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora_bias', type=str, default='none', help='LoRA bias')
    parser.add_argument('--lora_dropout', type=float, default=0.1, help='LoRA dropout')
    parser.add_argument('--target_modules', type=Optional[str], default=None, help='LoRA target modules')
    parser.add_argument('--checkpoint_dir', type=str, default='.', help='Checkpoint directory')
    parser.add_argument('--use_trained_tokenizer', type=bool, default=False, help='Use a trained tokenizer')
    return parser.parse_args()

def load_tweets(filename, label):
    with open(filename, 'r', encoding='utf-8') as f:
        return [(line.lower().rstrip(), label) for line in f]

def prepare_dataset(tweets, labels, tokenizer):
    dataset = Dataset.from_dict({"text": tweets, "label": labels})
    return dataset.map(lambda examples: tokenizer(examples["text"], padding="longest"), batched=True)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return evaluate.load("accuracy").compute(predictions=predictions, references=labels)

def set_seed(seed: int) -> None:
    import os
    import random

    import numpy as np

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")

def main():
    set_seed(42)
    
    args = get_args()

    if not args.disable_wandb:
        wandb.init(project="CIL-logs", config=vars(args), name=args.run_name, save_code=True)

    # Load data
    data = []
    if not args.subset:
        data.extend(load_tweets('twitter-datasets/train_neg_full.txt', 0))
        data.extend(load_tweets('twitter-datasets/train_pos_full.txt', 1))
    else:
        data.extend(load_tweets('twitter-datasets/train_neg.txt', 0))
        data.extend(load_tweets('twitter-datasets/train_pos.txt', 1))

    # Remove duplicates
    data = list(set(data))
    tweets, labels = zip(*data)

    # Tokenize and prepare datasets
    if not args.use_trained_tokenizer:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    else:
        tokenizer = AutoTokenizer.from_pretrained(f"tokenizers/{args.model_name}")
    if args.model_name == "microsoft/phi-2":
        tokenizer.pad_token = tokenizer.eos_token # yeah why would hf possibly do this for me

    dataset = prepare_dataset(tweets, labels, tokenizer)

    # Split dataset
    dataset = dataset.train_test_split(test_size=0.001, seed=42)
    if args.train_samples:
        dataset["train"] = dataset["train"].select(range(args.train_samples))
        dataset["test"] = dataset["test"]

    # Prepare model
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    if args.lora:
        config = LoraConfig(
            r=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias=args.lora_bias,
            target_modules=args.target_modules,
            modules_to_save=["classifier"],  # which weights not to freeze
            task_type=TaskType.SEQ_CLS,
        )
        model = get_peft_model(model, config)
    print("Trainable parameters ratio:", trained_params_ratio(model))

    output_dir = f"{args.checkpoint_dir}/checkpoints/{args.run_name}"

    # Prepare training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.n_epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_every,
        save_steps=args.eval_every,
        load_best_model_at_end=True,
        metric_for_best_model="accuracy",
        report_to="wandb" if not args.disable_wandb else "none",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train and evaluate
    trainer.train()
    final_eval = trainer.evaluate()
    print("Final evaluation:", final_eval)

    # Save best model
    trainer.save_model(f"{output_dir}/checkpoint_best")

if __name__ == "__main__":
    main()
