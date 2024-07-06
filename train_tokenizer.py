import argparse
from transformers import AutoTokenizer

def load_tweets(filename, label):
    with open(filename, 'r', encoding='utf-8') as f:
        return [(line.lower().rstrip(), label) for line in f]

def main():
    parser = argparse.ArgumentParser(description="Train a Hugging Face tokenizer on a tweet dataset.")
    parser.add_argument("tokenizer_name", type=str, help="Name of the tokenizer to train (e.g., 'bert-base-uncased')")
    args = parser.parse_args()

    # Load the dataset
    neg_tweets = load_tweets('twitter-datasets/train_neg_full.txt', 0)
    pos_tweets = load_tweets('twitter-datasets/train_pos_full.txt', 1)
    
    all_tweets = neg_tweets + pos_tweets
    tweets, labels = zip(*all_tweets)

    # Initialize the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)

    # Train the tokenizer
    def batch_iterator():
        for i in range(0, len(tweets), 1000):
            yield tweets[i:i+1000]

    new_tokenizer = tokenizer.train_new_from_iterator(batch_iterator(), vocab_size=30522)

    # Save the trained tokenizer
    save_name = f"tokenizers/{args.tokenizer_name}"
    new_tokenizer.save_pretrained(save_name)
    print(f"Trained tokenizer saved as '{save_name}'")

if __name__ == "__main__":
    main()