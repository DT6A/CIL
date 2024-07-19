import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import argparse


import numpy as np
import pandas as pd

# from vllm import LLM, SamplingParams


def get_args():
    parser = argparse.ArgumentParser(description='Tuning')
    parser.add_argument(
        '--predictions', type=str, default="test", help='Wandb run name'
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
                tweets.append(line.lower().rstrip())
                labels.append(label)

    load_tweets('twitter-datasets/test_data.txt', 0)
    # Convert to NumPy array to facilitate indexing
    tweets = np.array(tweets)
    labels = np.array(labels)

    original_df = pd.read_csv(args.predictions)

    result = original_df["Prediction"]
    swaps = 0
    new_preds = []
    for r, t in zip(result, tweets):
        opening = t.count("(")
        closing = t.count(")")
        if opening > closing and r == 1:
            swaps += 1
            print(t, r)
            new_preds.append(-1)
        elif opening < closing and r == -1:
            swaps += 1
            print(t, r)
            new_preds.append(1)
        else:
            new_preds.append(r)

    print("Number of swaps:", swaps)
    original_df["Prediction"] = new_preds
    # print(original_df)
    # id_column = list(range(1, 10001))
    # result = [2 * r - 1 for r in result]
    # df = pd.DataFrame({
    #     'Id': id_column,
    #     'Prediction': result
    # })
    original_df.to_csv(f'{args.predictions.split(".")[0]}-swapped.csv', index=False)
