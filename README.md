# CIL 2024 Text Classification
## Dependencies
- `torch` (for training and inference)
- `numpy` (for training and inference)
- `wandb` (for training)
- `pandas` (for inference)
- `transformers` (for training and inference)
- `evaluate` (for training and inference)
- `tqdm` (for training and inference)

## Hardware
We used `Quadro RTX 8000` with 48gb RAM. 

## Data
All scripts expect the data to be in the `twitter-datasets` folder

## Fine-tuning

Here is the table with fine-tuning results from the report and commands we used

| Name (with WandB link)                                                 | Command                                                                                                                                                           | Result  |
|------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|---------|
| [BERT-base, LoRA](https://wandb.ai/jovvik/CIL-logs/runs/txvqrleg)      | `python3 train_hf.py --batch_size 512 --run_name bert_lora --lora`                                                                                                | 0.85440 |
| [BERT-base](https://wandb.ai/jovvik/CIL-logs/runs/9c2799so)            | `python3 train_hf.py --batch_size 320 --run_name bert_full`                                                                                                         | 0.88840 |
| [BERT-large](https://wandb.ai/jovvik/CIL-logs/runs/283jicns)           | `python3 train_hf.py --batch_size 128 --run_name bert-large-uncased-full --model_name google-bert/bert-large-uncased`                                             | 0.89440 |
| [BERT-large, 4 epochs](https://wandb.ai/jovvik/CIL-logs/runs/lnnnrjnn) | `python3 train_hf.py --eval_every 2000 --run_name bert-large-uncased-full-4epoch-fixed --n_epochs 4 --batch_size 128 --model_name google-bert/bert-large-uncased` | 0.89200 |
| [FLAN-small](https://wandb.ai/jovvik/CIL-logs/runs/aad3y71h)           | `python3 train_hf.py --eval_every 2500 --batch_size 128 --run_name flan-small-full --model_name google/flan-t5-small`                                             | 0.86380 |
| [FLAN-base](https://wandb.ai/jovvik/CIL-logs/runs/33keipj1)            | `python3 train_hf.py --eval_every 2000 --batch_size 32 --run_name flan-base-full --model_name google/flan-t5-base`                                                | 0.89380 |
| [FLAN-large](https://wandb.ai/jovvik/CIL-logs/runs/8iktm5a6)           | `python3 train_hf.py --eval_every 8000 --batch_size 10 --run_name flan-large-full --model_name google/flan-t5-large`                                              | **0.90860** |



To generate submission file from fine-tuned model we used:

```bash
python3 inference.py --model_name <model_name>
```

## Prompting

Here is the table with commands and names of the corresponding runs from the report

| Name                                  | Command                                                | Score  |
|---------------------------------------|--------------------------------------------------------|--------|
| Nous-Hermes-2                         | `python3 llm_inference.py`                             | 0.6648 |
| Nous-Hermes-2 + CoT                   | `python3 llm_inference.py --cot`                       | 0.6894 |
| Nous-Hermes-2 + confidence            | `python3 llm_inference.py --probs`                     | 0.7184 |
| Nous-Hermes-2 + confidence + SC       | `python3 llm_inference.py --probs --n_repeats 3`       | 0.7232 |
| Nous-Hermes-2 + confidence + CoT      | `python3 llm_inference.py --probs --cot`               | 0.7632 |
| Nous-Hermes-2 + confidence + CoT + SC | `python3 llm_inference.py --probs --cot --n_repeats 3` | 0.7640 |

## Modifications
- To compute predictions using *ensembling* we used the following command: `python3 ensembling.py` after we generated submissions for all aforementioned models
- To compute *brackets-heuristic* we used the following command: `python3 pairing.py --predictions <prediction_file>` after we generated respective submission file. Entries with `brackets` and `":)" swap` suffixes were generated using this script and base submission file.
