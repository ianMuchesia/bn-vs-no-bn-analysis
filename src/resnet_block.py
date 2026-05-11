import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(ResidualBlock,self).__init__()
        
        #Block 1
        self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size=3,stride=stride,padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        
        
        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels or stride !=1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
            )
        
        
    def forward(self,x):
        
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        
        #Handle Dimension Mismatch
     
        identity = self.shortcut(identity)
            
            
            
        out = out + identity
        
        out = self.relu(out)
        
        
        
        return out
    
    
    
class ResidualBlockWithNoBn(nn.Module):
    def __init__(self,in_channels,out_channels,stride):
        super(ResidualBlockWithNoBn,self).__init__()
        
        #Block 1
        self.conv1 = nn.Conv2d(in_channels,out_channels, kernel_size=3,stride=stride,padding=1,bias=False)
        #self.bn1 = nn.BatchNorm2d(out_channels)
        
        
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1,bias=False)
        #self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU()
        
        
        self.shortcut = nn.Sequential()
        
        if in_channels != out_channels or stride !=1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)
            )
        
        
    def forward(self,x):
        
        identity = x
        
        out = self.conv1(x)
        #out = self.bn1(out)
        out = self.relu(out)
        
        
        out = self.conv2(out)
        #out = self.bn2(out)
        
        
        #Handle Dimension Mismatch
     
        identity = self.shortcut(identity)
            
            
            
        out = out + identity
        
        out = self.relu(out)
        
        
        
        return out
    
    