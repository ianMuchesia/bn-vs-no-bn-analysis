import torch

def validate(model, dataloader, criterion,device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    
    with torch.no_grad():
        # Turn off gradient tracking
        for images, labels in dataloader:
            images,labels = images.to(device),labels.to(device)
            
            outputs = model(images)
            
            loss = criterion(outputs,labels)
            
            val_loss += loss.item()
            
            
            _,predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            
            correct += (predicted==labels).sum().item()
            
            
        avg_loss = val_loss /len(dataloader)
        
        accuracy = 100 * correct /total
        
        return avg_loss,accuracy