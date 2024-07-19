# Code to inference Hermes with HF Transformers
# Requires pytorch, transformers, bitsandbytes, sentencepiece, protobuf, and flash-attn packages
import argparse

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import LlamaTokenizer
from transformers import MistralForCausalLM


def get_args():
    parser = argparse.ArgumentParser(description='Tuning')
    parser.add_argument(
        '--probs', type=str, default=None, help='Probabilities path'
    )
    parser.add_argument(
        '--save_name', type=str, help='Probabilities path'
    )
    parser.add_argument('--cot', action='store_true', default=False, help='Use COT')
    parser.add_argument('--n_repeats', default=1, type=int, help='Number of repeats for self-consistency')
    args = parser.parse_args()
    return args


def create_prompt(text, confidence=None, cot=False):
    text_confidence = ""
    if confidence is not None:
        text_confidence = f" It is {round(confidence * 100, 1)}% probability that :) was deleted."
    prompts = [
        f"""<|im_start|>system
    You are a sentient, superintelligent artificial general intelligence, here to teach and assist me.<|im_end|>
    <|im_start|>user
    You are given a tweet from which either :) or :( was removed. Your task is to output 1 if :) was removed or 0 if :( was removed.{text_confidence if confidence is not None else ''} {'Output only one number and nothing else.' if not cot else 'Provide short explaination of your decision and output a number at the end.'}

    Examples:

    Input: "I love sunny days"
    Output: {'Person is happy about sunny days, so :) fits here. Answer: ' if cot else ''}1

    Input: "It's been a rough day"
    Output: {'The day was hard and the user is probably tired, so :( is more suitable. Answer:' if cot else ''}0

    Tweet: {text}
    <|im_start|>assistant""",
    ]
    return prompts


if __name__ == "__main__":
    args = get_args()

    tokenizer = LlamaTokenizer.from_pretrained('NousResearch/Nous-Hermes-2-Mistral-7B-DPO', trust_remote_code=True)
    model = MistralForCausalLM.from_pretrained(
        "NousResearch/Nous-Hermes-2-Mistral-7B-DPO",
        torch_dtype=torch.float16,
        device_map="auto",
    )

    tweets = []
    labels = []

    def load_tweets(filename, label):
        with open(filename, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                tweets.append(line.lower().rstrip())
                labels.append(label)


    load_tweets('twitter-datasets/test_data.txt', 1)

    tweets = np.array(tweets)
    labels = np.array(labels)

    if args.probs is not None:
        probs_df = pd.read_csv('flan-large-full_probs.csv')
        probs = probs_df["Prediction"]
    else:
        probs = [None] * len(tweets)

    preds = []
    do_sample = args.n_repeats != 1
    temperature = 0.7 if do_sample else 0.0

    for (t, p) in tqdm(zip(tweets, probs)):
        count = 0
        prompts = create_prompt(t, confidence=p, cot=args.cot)
        input_ids = tokenizer(prompts[0], return_tensors="pt").input_ids.to("cuda")
        for _ in range(args.n_repeats):
            generated_ids = model.generate(input_ids, max_new_tokens=100, temperature=temperature, do_sample=do_sample,
                                           eos_token_id=tokenizer.eos_token_id)
            response = tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True,
                                        clean_up_tokenization_space=True)
            count += 0 if '0' in response[-10:] else 1
        preds.append(0 if count / args.n_repeats < 0.5 else 1)

    result = [2 * r - 1 for r in preds]
    id_column = list(range(1, len(result) + 1))

    df = pd.DataFrame({
        'Id': id_column,
        'Prediction': result
    })
    df.to_csv(args.save_name, index=False)
