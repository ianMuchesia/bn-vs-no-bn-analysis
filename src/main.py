import torch
import torch.optim as optim
import json
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import time




from src.cnn_with_batchnorm import CNNWithBatchNorm
from src.plain_cnn import PlainCNN
from src.resnet_without_batchnorm import ResNetWithoutBN
from src.resnet import FullResNet
from torch.utils.data import Subset
from src.train_variations import training_model






def build_model(config):
    if config == "plain":
        return PlainCNN()

    elif config == "bn":
        return CNNWithBatchNorm()

    elif config == "skip":
        return ResNetWithoutBN()

    elif config == "resnet":
        return FullResNet()
    
    
    
    #1. Define transform ( Turn images into tensors )
transform = transforms.Compose([transforms.ToTensor(),transforms.RandomHorizontalFlip()])



#2. Download/Load datasets
train_dataset = torchvision.datasets.CIFAR10(root="./../experiments/data",train=True,download=True, transform=transform)


test_dataset = torchvision.datasets.CIFAR10(root="./../experiments/data",train=False,download=True, transform=transform)

# 2.5 Create a subset of 1000 images
subset_indices = list(range(10000))
train_subset = Subset(train_dataset, subset_indices)
test_subset = Subset(test_dataset,subset_indices)


#3. Create loaders
train_loader = DataLoader(train_subset,batch_size=64,shuffle=True)
test_loader = DataLoader(test_subset,batch_size=64,shuffle=False)


#4. Sanity check
data_iter = iter(train_loader)
images,labels = next(data_iter)



#5.Device check
device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')


print(f"Batch Images Shape: {images.shape}")
#(batch_size, channels, height,width)

print(f"Batch Labels Shape: {labels.shape}")





if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    print(f"Executing on device: {device}")
    
    configs_to_test = ["plain", "bn", "skip", "resnet"]
    
    for config in configs_to_test:
        # Build the specific architecture
        model = build_model(config)
        
        # Execute the unified pipeline
        training_model(model, config, train_loader, test_loader, device)