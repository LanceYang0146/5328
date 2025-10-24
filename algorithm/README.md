# Advanced ML A2 — Code (Label Noise Robust Classification)

This folder contains the **code-only** submission for COMP4328/5328 Assignment 2.

## Methods
1. **Forward Loss Correction** (`--method forward-knownT` and `--method forward-estT`)
   - Uses the transition matrix **T** to correct the loss ("forward correction").
   - For `forward-knownT`: loads the provided T for FashionMNIST0.3 & 0.6.
   - For `forward-estT`: estimates `T_hat` on CIFAR via a confident-anchor procedure.
2. **Generalized Cross Entropy (GCE)** (`--method gce`)
   - A robust loss that does not need T; works under various noise types.

## Datasets
Place the three `.npz` files in `../data/` (same level as this `algorithm/`), e.g.:
```
ROOT/
 ├─ algorithm/
 └─ data/
     ├─ FashionMNIST0.3.npz
     ├─ FashionMNIST0.6.npz
     └─ CIFAR.npz
```
**Do not** include the `.npz` files in your zip submission (this `data/` stays empty).

## Quickstart
```bash
cd algorithm
pip install -r requirements.txt

# One run on FashionMNIST0.3 with known T (forward correction)
python train.py --dataset FashionMNIST0.3 --method forward-knownT --epochs 10

# One run on CIFAR with T estimation + forward correction
python train.py --dataset CIFAR --method forward-estT --epochs 15

# One run on FashionMNIST0.6 with GCE loss
python train.py --dataset FashionMNIST0.6 --method gce --epochs 10

# 10 random splits, report mean/std (required in spec)
python run_experiments.py --dataset FashionMNIST0.3 --method forward-knownT --repeats 10
```

## Outputs
- Metrics are printed to stdout, and per-run logs are saved to `./runs/<timestamp>/...`

## Reproducibility
- We fix seeds per run, but **each repeat uses a different seed** to comply with the assignment.

## Files
- `datasets.py` : loads `.npz` datasets, creates random 80/20 train/val splits
- `models.py` : simple CNN backbones for 28x28 (grayscale) and 32x32x3 (RGB)
- `losses.py` : forward-corrected CE loss, GCE loss
- `estimate_T.py` : confident-anchor estimator for `T`
- `train.py` : training script for a single run
- `run_experiments.py` : loops `train.py` for N repeats and aggregates mean/std
- `utils.py` : helpers (seeding, accuracy, logging)

---

**Note:** Code aims for clarity and reasonable runtime; tune epochs/batch size if needed.
