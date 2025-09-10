# -*- coding: utf-8 -*-
# Bactrian-X
# - Paper: https://doi.org/10.48550/arXiv.2305.15011
# - Data: https://huggingface.co/datasets/MBZUAI/Bactrian-X
import pandas as pd
from datasets import load_dataset


# Load datasets
datasets = {
    "de": load_dataset("MBZUAI/Bactrian-X", "de")['train'],
    "fr": load_dataset("MBZUAI/Bactrian-X", "fr")['train'],
    "es": load_dataset("MBZUAI/Bactrian-X", "es")['train'],
    "it": load_dataset("MBZUAI/Bactrian-X", "it")['train'],
    "nl": load_dataset("MBZUAI/Bactrian-X", "nl")['train'],
    "pt": load_dataset("MBZUAI/Bactrian-X", "pt")['train'],
    "ru": load_dataset("MBZUAI/Bactrian-X", "ru")['train'],
    "cs": load_dataset("MBZUAI/Bactrian-X", "cs")['train'],
    "pl": load_dataset("MBZUAI/Bactrian-X", "pl")['train'],
    "ar": load_dataset("MBZUAI/Bactrian-X", "ar")['train'],
    "fa": load_dataset("MBZUAI/Bactrian-X", "fa")['train'],
    "he": load_dataset("MBZUAI/Bactrian-X", "he")['train'],
    "tr": load_dataset("MBZUAI/Bactrian-X", "tr")['train'],
    "ja": load_dataset("MBZUAI/Bactrian-X", "ja")['train'],
    "ko": load_dataset("MBZUAI/Bactrian-X", "ko")['train'],
    "vi": load_dataset("MBZUAI/Bactrian-X", "vi")['train'],
    "th": load_dataset("MBZUAI/Bactrian-X", "th")['train'],
    "id": load_dataset("MBZUAI/Bactrian-X", "id")['train'],
    "my": load_dataset("MBZUAI/Bactrian-X", "my")['train'],
    "km": load_dataset("MBZUAI/Bactrian-X", "km")['train'],
    "tl": load_dataset("MBZUAI/Bactrian-X", "tl")['train'],
    "hi": load_dataset("MBZUAI/Bactrian-X", "hi")['train'],
    "bn": load_dataset("MBZUAI/Bactrian-X", "bn")['train'],
    "ur": load_dataset("MBZUAI/Bactrian-X", "ur")['train'],
}

# Convert datasets to pandas DataFrame
dfs = {lang: pd.DataFrame(dataset) for lang, dataset in datasets.items()}

# Remove rows where 'input' column is not empty
for lang, df in dfs.items():
    dfs[lang] = df[df['input'] == '']
    
# Sample 800 unique ids from each dataset
sampled_dfs = {}
all_ids = set()

for lang, df in dfs.items():
    unique_df = df[~df['id'].isin(all_ids)]
    sampled_df = unique_df.sample(n=800, random_state=1)
    all_ids.update(sampled_df['id'])
    sampled_dfs[lang] = sampled_df

# Verify the sampled data
for lang, df in sampled_dfs.items():
    print(f"{lang}: {df.shape}")

combined_df = pd.concat(sampled_dfs.values(), ignore_index=True)
combined_df = combined_df[['instruction', 'output']]
combined_df = combined_df.sample(frac=1)
combined_df.to_parquet("../../data/Bactrian-X/bactrian_sample.parquet", index=False)