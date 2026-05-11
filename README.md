# BN vs No-BN Analysis
Short study of batch normalization vs no-batchnorm in CNNs and ResNets on CIFAR-10.

## Goal
- compare training dynamics with and without batch normalization
- contrast plain CNN and ResNet variants
- track gradient norms across layers

## Models
- Plain CNN
- CNN + BatchNorm
- ResNet
- ResNet without BatchNorm

## Data
- CIFAR-10 with basic augmentation
- optional subset for quick experiments

## Project layout
- [src/](src/) model, training, and evaluation code
- [notebooks/](notebooks/) analysis notebooks
- [experiments/](experiments/) logs and artifacts
- [REPORT.md](REPORT.md) write-up

## Run
- install deps: torch, torchvision
- run training: python -m src.train_variations
- logs are written to [experiments/](experiments/) as JSON
## Notes
- gradient hooks record per-epoch norms for key layers