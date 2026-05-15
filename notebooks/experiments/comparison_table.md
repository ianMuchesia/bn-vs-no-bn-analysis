Model | Final Val Acc | Train Time | Stability / Behavior
---|---|---|---
Plain | 48.5% | 794s | Flatlined / Gradients died
BatchNorm | 71.2% | 854s | Steady learning, batch noise
Skip | 61.3% | 768s | Flatlined / Minor improvement
ResNet | 79.9% | 827s | High capacity, severe overfitting
