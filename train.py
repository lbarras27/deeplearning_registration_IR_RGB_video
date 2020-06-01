import torch
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
    This loss penalize if the transformation is not smooth.
    
    @s: the flow that we want compute the loss.
"""
def gradient_loss(s, penalty='l2'):
    dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :]) 
    dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1]) 

    if(penalty == 'l2'):
        dy = dy * dy
        dx = dx * dx

    d = torch.mean(dx) + torch.mean(dy)
    return d / 2.0

"""
    Train the Netmain or the AffineNet network.
    
    @model: the model that we want to train.
    @fixed_input: the image(s) that is stay fixed (rgb image(s) in our case).
    @moving_input: the image(s) we want to align to the fixed_input (ir images in our case).
    @epoch: the number of epochs.
    @mini_batch_size: the size of the batch.
    @criterion: the loss we want to use (MSE for example).
    @optimizer: the algorithm used to update parameters (Adam for example).
    @type_model: the network name we use ('affine1' or 'netMain')
    @reg: the regularization parameter for the loss function (if Netmain network is used)
"""
def train(model, fixed_input, moving_input, epoch, mini_batch_size, criterion, optimizer, type_model='affine1', reg=0.2):
    model.train()
    for i in range(1, epoch+1):
        sum_loss = 0
        for b in range(0, fixed_input.size(0), mini_batch_size):
            
            optimizer.zero_grad()
            if type_model=='affine1':
                warp, flow = model(moving_input.narrow(0, b, mini_batch_size))
            else:
                warp, flow = model(moving_input.narrow(0, b, mini_batch_size), fixed_input.narrow(0, b, mini_batch_size))
                
            recon_loss = criterion(warp, fixed_input.narrow(0, b, mini_batch_size))
            if type_model=='affine1':
                loss = recon_loss
            else:
                grad_loss = gradient_loss(flow)
                loss = recon_loss + reg * grad_loss
            
            sum_loss = sum_loss + loss.item()
            
            loss.backward()
            optimizer.step()

        print('epoch ' + str(i) + ': loss =', sum_loss)
        
"""
    Train the Netmain or the AffineNet network. The difference with train() is that we not need to load all the inputs images 
    on the gpu memories before to call this function. (useful if not enough memory on the gpu to load all the inputs images.
    
    @model: the model that we want to train.
    @fixed_input: the image(s) that is stay fixed (rgb image(s) in our case).
    @moving_input: the image(s) we want to align to the fixed_input (ir images in our case).
    @epoch: the number of epochs.
    @mini_batch_size: the size of the batch.
    @criterion: the loss we want to use (MSE for example).
    @optimizer: the algorithm used to update parameters (Adam for example).
    @type_model: the network name we use ('affine1' or 'netMain')
    @reg: the regularization parameter for the loss function (if Netmain network is used)
"""
def train2(model, fixed_input, moving_input, epoch, mini_batch_size, criterion, optimizer, type_model='affine1', reg=0.2):
    model.train()
    for i in range(1, epoch+1):
        sum_loss = 0
        for b in range(0, fixed_input.size(0), mini_batch_size):
            
            optimizer.zero_grad()
            if type_model=='affine1':
                warp, flow = model(moving_input.narrow(0, b, mini_batch_size).to(device))
            else:
                warp, flow = model(moving_input.narrow(0, b, mini_batch_size), fixed_input.narrow(0, b, mini_batch_size))
                
            recon_loss = criterion(warp, fixed_input.narrow(0, b, mini_batch_size).to(device))
            if type_model=='affine1':
                loss = recon_loss
            else:
                grad_loss = gradient_loss(flow)
                loss = recon_loss + reg * grad_loss
            
            sum_loss = sum_loss + loss.item()
            
            loss.backward()
            optimizer.step()

        print('epoch ' + str(i) + ': loss =', sum_loss)