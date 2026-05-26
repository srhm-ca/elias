# Data for "Elias in the Lighthouse, Again?"
This repository contains IDs for stories in OLMo 3's post-training set.

## Released File

`dolci_record_ids.csv` contains:

| Column | Description |
|---|---|
| `dolci_file` | Which post-training split the record belongs to |
| `hf_dataset` | Hugging Face dataset name for reconstruction |
| `id_column` | Which column in the HF dataset holds the matching identifier |
| `id` | The record identifier (matches `id_column` in the HF dataset) |

## Dolci Datasets on Hugging Face

| dolci_file | HF Dataset | ID Column | HF Rows |
|---|---|---|---|
| `dolci-instruct-sft` | `allenai/dolci-instruct-sft` | `id` | 2,152,112 |
| `dolci-instruct-dpo` | `allenai/dolci-instruct-dpo` | `prompt_id` | 260,000 |
| `dolci-instruct-rl` | `allenai/dolci-instruct-rl` | `id` | 170,000 |
| `dolci-think-sft-7b` | `allenai/dolci-think-sft-7b` | `id` | ~1.7M |
| `dolci-think-dpo-7b` | `allenai/dolci-think-dpo-7b` | `id` | 150,000 |
| `dolci-think-rl-7b` | `allenai/dolci-think-rl-7b` | `custom_id` | 102,014 |

All datasets are publicly available under open licenses (ODC-BY, Apache 2.0, or MIT).

## Reconstruction

```bash
pip install datasets pandas
```

```python
import pandas as pd
from datasets import load_dataset

ids_df = pd.read_csv("dolci_record_ids.csv")

for hf_name, group in ids_df.groupby("hf_dataset"):
    id_col = group["id_column"].iloc[0]
    target_ids = set(group["id"])

    ds = load_dataset(hf_name, split="train")

    def keep(record):
        return str(record[id_col]) in target_ids

    filtered = ds.filter(keep)
    print(f"{hf_name}: matched {len(filtered)} / {len(target_ids)} IDs")

    # Extract text from the filtered records
    # (text extraction varies by dataset, see below)
```

The original text field varies by dataset structure:

| dolci_file | How to extract text |
|---|---|
| `dolci-instruct-sft` | `record["messages"][1]["content"]` (assistant response) |
| `dolci-instruct-dpo` | `record["chosen"][1]["content"]` (chosen response) |
| `dolci-instruct-rl` | `record["prompt"]` |
| `dolci-think-sft-7b` | `record["messages"][1]["content"]` |
| `dolci-think-dpo-7b` | `record["chosen"][1]["content"]` |
| `dolci-think-rl-7b` | `" ".join(record["outputs"])` |
