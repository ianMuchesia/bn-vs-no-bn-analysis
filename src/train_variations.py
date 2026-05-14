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
from src.gradient_hook import get_gradient_hook
from src.validation import validate







def training_model(model,config_name,train_loader,test_loader,device):
    
    
    print(f"\n--- Starting Experiment : {config_name.upper()} ---")
    
    
    #1. Hardware Mapping
    model = model.to(device)

    criterion = torch.nn.CrossEntropyLoss() 
    optimizer = optim.SGD(model.parameters(),lr=0.001,momentum=0.9,weight_decay=1e-4)


    #Training Loop
    epochs = 15
    best_accuracy = 0.0
    best_val_loss = float('inf')
    history = []
    checkpoint_dir = "experiments/run_latest"

    epoch_gradient_history = {}
    
    target_layers = [ 'conv1.weight','conv2.weight', 'conv3.weight','conv4.weight','fc.weight']
    
    
    for layer in target_layers:
        epoch_gradient_history[layer] = []
    
    starting_time = time.perf_counter()


    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        
        # Accumulators for gradient magnitudes per layer for this specific epoch
        grad_accumulators_magnitude = {layer: 0.0 for layer in target_layers}
        grad_accumulators_mean = {layer: 0.0 for layer in target_layers}
        grad_accumulators_var = {layer: 0.0 for layer in target_layers}
        
        epoch_start_time = time.perf_counter()
        
        
        for i,(images,labels) in enumerate(train_loader):
            
            images,labels = images.to(device),labels.to(device)
            #Zero gradient
            optimizer.zero_grad()
            
            
            #Forward Pass
            outputs = model(images)
            
            loss = criterion(outputs,labels)
            
            
            
            #Backward Pass + update
            loss.backward()
            
            #GRADIENT TRACKING
            for name, param in model.named_parameters():
                if name in target_layers and param.grad is not None:
                    #Calculate the L2 norm (magnitude) of the gradient tensor
                    grad_magnitude = param.grad.data.norm(2).item()
                    grad_mean = param.grad.data.mean().item()
                    grad_var = param.grad.data.var().item()
                    grad_accumulators_magnitude[name] += grad_magnitude
                    
                    grad_accumulators_mean[name] += grad_mean
                    grad_accumulators_var[name] += grad_var
                
            optimizer.step()
            
            
            #Metrics Calculation
            running_loss += loss.item()
            
            
            _,predicted = torch.max(outputs,1)
            
            total += labels.size(0)
            
            
            correct += (predicted == labels).sum().item()
            
            
        #Print stats per epoch
        train_accuracy = 100 * correct/total
        
        avg_train_loss = running_loss / len(train_loader)
        
        print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {running_loss/len(train_loader):.4f} -Training Accuracy: {train_accuracy:.2f}%")
        
        epoch_grads = {}
        for layer in target_layers:
            avg_magnituede_grad = grad_accumulators_magnitude[layer] / len(train_loader)
            avg_mean_grad = grad_accumulators_mean[layer] / len(train_loader)
            avg_var_grad = grad_accumulators_var[layer] /len(train_loader)
            grads = {"magnitude":avg_magnituede_grad,"mean":avg_mean_grad,"var":avg_var_grad}
            epoch_gradient_history[layer].append(grads)
            epoch_grads[layer] = grads
            
            
        #Validation Phase
        #Assuming you validate() function sets model.eval() and uses torch.no_grad()      
        avg_val_loss, val_accuracy = validate(model, test_loader,criterion,device)
        
        
        #Temporal Tracking Fix
        epoch_end_time = time.perf_counter()
        elapsed_time = epoch_end_time - epoch_start_time
        
        print(f"Epoch [{epoch + 1}/{epochs}] | Train Loss:{avg_train_loss:.4f} | Train Acc: {train_accuracy:.2f} | Val Acc: {val_accuracy:.2f} | Time: {elapsed_time}s")
        
        
        #Track Best States
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            
        #Logging 
        epoch_metrics = {
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_acc":train_accuracy,
            "val_loss": avg_val_loss,
            "val_acc":val_accuracy,
            "time_taken":elapsed_time
        }
            
        
        history.append(epoch_metrics)
        
       
        
    with open(f"./experiments/training_{config_name}_log.json","w") as f:
        json.dump(history, f, indent=4)
    
    with open(f"./experiments/{config_name}_gradient_log.json","w") as f:
        json.dump(epoch_gradient_history, f, indent=4)
        
        
    total_time = time.perf_counter() - starting_time
    print(f"Completed {config_name} in {total_time:.2f} seconds. Best Val Acc: {best_accuracy:.2f}%\n")
    
    return history, epoch_gradient_history
        
        
    
    
