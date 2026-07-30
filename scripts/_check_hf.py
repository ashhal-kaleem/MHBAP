"""Quick HuggingFace dataset availability check."""
import sys
print("Starting HF check...", flush=True)

from datasets import load_dataset

datasets_to_check = [
    ("deanngkl/raf-db-7emotions", "train", "RAF-DB"),
    ("LouisSimon/wesad-parquet", "train", "WESAD"),
]

for hf_id, split, name in datasets_to_check:
    try:
        ds = load_dataset(hf_id, split=split, streaming=True, trust_remote_code=False)
        item = next(iter(ds))
        print(f"{name} OK: keys={list(item.keys())}", flush=True)
    except Exception as e:
        print(f"{name} FAIL: {repr(e)[:300]}", flush=True)

print("Done.", flush=True)
