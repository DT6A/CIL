import pandas as pd


if __name__ == "__main__":
    tables = [
        "bert-base-uncased-lora-nopad",
        "bert-large-uncased-full",
        "bert-large-uncased-full-4epoch-fixed",
        "bert_full",
        "flan-base-full",
        "flan-large-full",
        "flan-small-full",
    ]

    sum_predictions = None
    count = 0

    for file in tables:
        df = pd.read_csv(f"predictions/{file}.csv")
        if 'Prediction' in df.columns:
            if sum_predictions is None:
                sum_predictions = df[['Prediction']].copy()
            else:
                sum_predictions['Prediction'] += df['Prediction']
            count += 1

    average_predictions = sum_predictions / count
    predictions = [1 if p >= 0 else -1 for p in average_predictions['Prediction']]

    df['Prediction'] = predictions
    df.to_csv("predictions/ensemble.csv", index=False)
