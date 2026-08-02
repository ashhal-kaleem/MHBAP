import os
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
from datasets import load_dataset

print('=== FER2013 (clip-benchmark) ===')
try:
    ds = load_dataset('clip-benchmark/wds_fer2013', split='test[:3]')
    print('keys:', list(ds[0].keys()))
    print('cls sample:', ds[0]['cls'], ds[1]['cls'], ds[2]['cls'])
    img = ds[0]['jpg']
    print('image type:', type(img), 'size:', img.size if hasattr(img, 'size') else len(img))
    # class distribution in train
    train_ds = load_dataset('clip-benchmark/wds_fer2013', split='train[:100]')
    labels = [x['cls'] for x in train_ds]
    from collections import Counter
    print('train label dist (100):', Counter(labels))
except Exception as e:
    print('FER ERR:', e)

print()
print('=== WESAD parquet ===')
try:
    wds = load_dataset('LouisSimon/wesad-parquet', split='train[:3]')
    print('keys:', list(wds[0].keys()))
    print('sample0:', {k: str(v)[:60] for k, v in wds[0].items()})
except Exception as e:
    print('WESAD ERR:', e)
