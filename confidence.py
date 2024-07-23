import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse


import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# from vllm import LLM, SamplingParams


def get_args():
    parser = argparse.ArgumentParser(description='Tuning')
    parser.add_argument(
        '--probs', type=str, default="test", help='Probabilities CSV'
    )
    parser.add_argument(
        '--predictions', type=str, default="test", help='LLM predictions CSV'
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    tweets = []
    labels = []


    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                tweets.append(" ".join(line.lower().rstrip().split(',')[1:]))
                labels.append(label)

    load_tweets('twitter-datasets/test_data.txt', 0)

    tweets = np.array(tweets)
    labels = np.array(labels)

    probs_df = pd.read_csv(args.probs)
    preds_df = pd.read_csv(args.predictions)

    probs = probs_df["Prediction"]
    preds = preds_df["Prediction"]

    only_positive_changes = []
    ignored_probs = []
    for p, pred in zip(probs, preds):
        if p >= 0.5 and pred == -1 or p < 0.5 and pred == 1:
            ignored_probs.append(p)
        if p < 0.5:
            only_positive_changes.append(-1)
        else:
            only_positive_changes.append(1 if pred == 1 else -1)

    plt.hist(probs, edgecolor='black', label="Confidences")
    plt.hist(ignored_probs, edgecolor='black', label="Ignored confidence")
    plt.xlabel("Prediction confidence")
    plt.ylabel("Number of samples")
    plt.title("Distribution of confidences for the fine-tuned model and ignored confidences")
    plt.legend()
    plt.grid()
    plt.savefig("confidences_distr.pdf", dpi=300, bbox_inches='tight')
    probs_df["Prediction"] = only_positive_changes
    print("Swapped predictions:", len(ignored_probs))


    # plt.xlabel("Prediction confidence")
    # plt.ylabel("Number of samples")
    # plt.title("Distribution of the ignored confidences")
    # plt.savefig("ignored.pdf", dpi=300, bbox_inches='tight')
    # plt.show()

    probs_df.to_csv(f'{args.probs.split(".")[0]}_only_pos_swap.csv', index=False)
