#!/usr/bin/env bash
set -e
# Example batch commands
python train.py --dataset FashionMNIST0.3 --method forward-knownT --epochs 10
python train.py --dataset FashionMNIST0.6 --method gce --epochs 10
python train.py --dataset CIFAR --method forward-estT --epochs 15
