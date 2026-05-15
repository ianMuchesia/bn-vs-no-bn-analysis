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
- [REPORT.md](REPORT.md) final write-up

## Run
- install deps: torch, torchvision
- run training: python -m src.train_variations
- logs are written to [experiments/](experiments/) as JSON

## Key findings (final)
- BatchNorm is the primary enabler for learning in these models.
- Skip connections without normalization provide only minor gains.
- ResNet yields the highest accuracy but shows strong overfitting.

## Results snapshot
Model | Final Val Acc | Train Time | Stability / Behavior
---|---|---|---
Plain | 48.5% | 794s | Flatlined / Gradients died
BatchNorm | 71.2% | 854s | Steady learning, batch noise
Skip | 61.3% | 768s | Flatlined / Minor improvement
ResNet | 79.9% | 827s | High capacity, severe overfitting

## Notes
- gradient hooks record per-epoch norms for key layers
- see [REPORT.md](REPORT.md) for full analysis narrative