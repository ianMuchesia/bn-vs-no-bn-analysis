




def get_gradient_hook(layer_name,gradients):
    #a hook to store the norm(magnitude) of the gradient.
    
    def hook(module,grad_input,grad_output):
        
        gradients[layer_name] = grad_output[0].detach().norm().item()
        
    return hook