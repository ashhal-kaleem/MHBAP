import torch
import sys
import json

sys.path.append(r"d:\MHBAP")

def main():
    ckpt_path = r"d:\MHBAP\ml\models\weights\tcmt_trained.pt"
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu")
        print("Keys in checkpoint:", ckpt.keys())
        for k in ["datasets", "n_train", "n_val", "n_test"]:
            if k in ckpt:
                print(f"{k}: {ckpt[k]}")
        if "test_metrics" in ckpt:
            print("test_metrics in checkpoint:")
            print(json.dumps(ckpt["test_metrics"], indent=2))
    except Exception as e:
        print("Error loading checkpoint:", e)

if __name__ == "__main__":
    main()
