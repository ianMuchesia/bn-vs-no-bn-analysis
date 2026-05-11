import torch.nn as nn
from src.resnet_block import ResidualBlock
class FullResNet(nn.Module):
    
    def __init__(self,num_classes=10):
        super(FullResNet, self).__init__()
        #1. Initial layer
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=16,kernel_size=3,stride=1,bias=False)
        
        self.bn1 = nn.BatchNorm2d(num_features=16)
        self.relu = nn.ReLU()
        
        
        #2. Residual Blocks
        self.conv2 = ResidualBlock(16,32,1)
        self.conv3 = ResidualBlock(32,64,2)
        self.conv4 = ResidualBlock(64,128,2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=128,out_features=num_classes)
        
        
        
    def forward(self,x):
        
        out = self.conv1(x)
        
        out = self.bn1(out)
        
        out = self.relu(out)
        
        out = self.conv2(out)
        
        out = self.conv3(out)
        
        out = self.conv4(out)
        
     
        
      
        
        
        
        out = self.avgpool(out)
        
        out = self.flatten(out)
        
        out = self.fc(out)
        
        return out