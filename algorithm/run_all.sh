#!/usr/bin/env bash
set -e

# =======================================
# FASHION-MNIST (Noise = 0.3)
# =======================================

# Forward-knownT (baseline with true T)
!python run_experiments.py --dataset FashionMNIST0.3 --method forward-knownT --repeats 10 --epochs 30 --csv results.csv
!cp summary.json /content/drive/MyDrive/5328/summary_FMNIST03_forward.json

# Forward-estT (estimate T̂ + report RRE) ← recommended new variant
!python run_experiments.py --dataset FashionMNIST0.3 --method forward-estT --repeats 10 --epochs 30 --csv results.csv
!cp summary.json /content/drive/MyDrive/5328/summary_FMNIST03_forward_estT.json

# GCE (robust loss baseline, independent of T̂)
!python run_experiments.py --dataset FashionMNIST0.3 --method gce --repeats 10 --epochs 30 --csv results.csv
!cp summary.json /content/drive/MyDrive/5328/summary_FMNIST03_gce.json

# =======================================
# FASHION-MNIST (Noise = 0.6)
# =======================================

# Forward-knownT (baseline with true T)
!python run_experiments.py --dataset FashionMNIST0.6 --method forward-knownT --repeats 10 --epochs 30 --csv results.csv
!cp summary.json /content/drive/MyDrive/5328/summary_FMNIST06_forward.json

# Forward-estT (estimate T̂ + report RRE)
!python run_experiments.py --dataset FashionMNIST0.6 --method forward-estT --repeats 10 --epochs 30 --csv results.csv
!cp summary.json /content/drive/MyDrive/5328/summary_FMNIST06_forward_estT.json

# GCE (robust loss baseline, independent of T̂)
!python run_experiments.py --dataset FashionMNIST0.6 --method gce --repeats 10 --epochs 30 --csv results.csv
!cp summary.json /content/drive/MyDrive/5328/summary_FMNIST06_gce.json

# =======================================
# CIFAR (Unknown T — estimate T̂)
# =======================================

# Forward-estT (main: estimate T̂ then forward correction)
!python run_experiments.py --dataset CIFAR --method forward-estT --repeats 10 --epochs 80 --csv results.csv
!cp summary.json /content/drive/MyDrive/5328/summary_CIFAR_forward_estT.json

# GCE baseline (no T̂ dependency, for comparison)
!python run_experiments.py --dataset CIFAR --method gce --repeats 10 --epochs 80 --csv results.csv
!cp summary.json /content/drive/MyDrive/5328/summary_CIFAR_gce.json


# =======================================
# Final summary table
# =======================================
!cp results.csv /content/drive/MyDrive/5328/results_all.csv
