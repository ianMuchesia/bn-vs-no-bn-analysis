import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNWithBatchNorm(nn.Module):
    def __init__(self):
        super(CNNWithBatchNorm,self).__init__()
        
        #Block 1
        self.conv1 = nn.Conv2d(3,16,kernel_size=3,padding=1,stride=1)
        self.bn1 = nn.BatchNorm2d(16)
        
        #Block2
        self.conv2 = nn.Conv2d(16,32,kernel_size=3,padding=1,stride=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        #Block3
        self.conv3 = nn.Conv2d(32,64,kernel_size=3,padding=1,stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        #Block4
        self.conv4 = nn.Conv2d(64,128,kernel_size=3,padding=1,stride=1)
        self.bn4 = nn.BatchNorm2d(128)
       
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        
        
        #Dense Layer
        #self.fc1 = nn.Linear(512,128)
        self.fc = nn.Linear(128,10)
        
        
        #Dropout to reduce overfitting
        self.dropout = nn.Dropout(p=0.5)
        
        
    
    def forward(self,x):
        
       
        
        Z1 = self.conv1(x)
        
        A = F.relu(Z1)
        
        P1 = self.bn1(A)
        
        Z2 = self.conv2(P1)
        
        A2 = F.relu(Z2)
        
        P2 = self.bn2(A2)
        
        Z3 = self.conv3(P2)
        
        A3 = F.relu(Z3)
        
        P3 = self.bn3(A3)
        
        
        Z4 = self.conv4(P3)
        
        A4 = F.relu(Z4)
        
        P4 = self.bn4(A4)
        
        
        out = self.avgpool(P4)
        
        
        
        X = torch.flatten(out,1)
        
        #Z5 = self.fc1(X)
        #Z5 = self.dropout(X)
        A5 = F.relu(X)
        Z6 = self.fc(A5)
        
        return Z6