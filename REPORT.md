# FINAL ANALYSIS REPORT
### Setup
- CIFAR-10 with basic augmentation

### Architectures
- Plain CNN
- CNN + BatchNorm
- ResNet
- ResNet without BatchNorm


### Findings
Model | Final Val Acc | Train Time | Stability / Behavior
---|---|---|---
Plain | 48.5% | 794s | Flatlined / Gradients died
BatchNorm | 71.2% | 854s | Steady learning, batch noise
Skip | 61.3% | 768s | Flatlined / Minor improvement
ResNet | 79.9% | 827s | High capacity, severe overfitting



### Does BatchNorm help?
Yes. It is the absolute gatekeeper for learning in deep networks. Without it, the loss flatlines at 2.3 (random guessing). It adds computational time per epoch, but it prevents the gradients from vanishing, allowing the network to actually update its weights.

### Do Skip Connections help?
On their own (without BN), they barely do anything in this specific architecture. They provide a slight boost over the Plain CNN, but the model still fails to learn meaningful features.


### Which matters more?
BatchNorm matters more. The data proves it. The CNN + BatchNorm model actually learned and dropped its loss, while the CNN + Skip model completely flatlined. Skip connections are useless if the data flowing through them isn't normalized.


### Combined Effect (ResNet):
When combined, ResNet provides the highest learning capacity. However, the empirical data shows this extreme capacity leads directly to severe overfitting if not controlled by regularization. It does not produce the "most stable" visual curve (due to batch noise), but it absolutely produces the highest peak performance.