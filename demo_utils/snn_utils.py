from torch import nn
import torch
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
from snntorch import utils
from snntorch import functional as SF
import snntorch as snn
import pandas as pd
import os

class RateEncodingLayer(nn.Module):
    def __init__(self, num_steps=10):
        super().__init__()
        self.num_steps=num_steps
    def forward(self, images): # self-made layer, forward arrguments only contain x
        batch_size, _, height, width = images.shape
        images = images.view(batch_size, -1)  # [batch_size, 784]
        spike_sequence = torch.bernoulli(images.unsqueeze(0).repeat(self.num_steps, 1, 1))  # [time_steps, batch_size, 784]
        return spike_sequence.view(self.num_steps, batch_size, height*width)
       

def rate_encoding_images(images, time_steps=10):
    """
    Encode images to spike train and reshape.

    Args:
        images (list): 
            [batch_size, 1, 28, 28]
        time_steps (int): 
            Number of time steps
    Returns: 
        lists:
            [time_steps, batch_size, 784]
    """
    # images, _ = next(iter(data_loader))
    batch_size, _, height, width = images.shape
    images = images.view(batch_size, -1)  # [batch_size, 784]
    spike_sequence = torch.bernoulli(images.unsqueeze(0).repeat(time_steps, 1, 1))  # [time_steps, batch_size, 784]
    return spike_sequence.view(time_steps, batch_size, height*width)

# def forward_pass(net, num_steps, data):
#     """
#     Pass data into network over time
#     """
#     mem_rec = []
#     spk_rec = []
#     utils.reset(net)  # resets hidden states for all LIF neurons in net

#     for step in range(num_steps):
#         spk_out, mem_out = net(data)
#         spk_rec.append(spk_out)
#         mem_rec.append(mem_out)

#     return torch.stack(spk_rec), torch.stack(mem_rec)

class forward_pass():
    """
    For the net that doesn't contain Time Iteration.

        spkin: The input is already coded spikes.

        valin: The input is value.
    """
    def __init__(self,net,num_steps,data):
        self.mem_rec = []
        self.spk_rec = []
        self.num_steps=num_steps
        self.data=data
        utils.reset(net)  # resets hidden states for all LIF neurons in net
        self.net=net
    def spkin(self):
        for step in range(self.num_steps):
            spk_out, mem_out = self.net(self.data[step])
            self.spk_rec.append(spk_out)
            self.mem_rec.append(mem_out)

        return torch.stack(self.spk_rec), torch.stack(self.mem_rec)
    def valin(self):
        for step in range(self.num_steps):
            spk_out, mem_out = self.net(self.data)
            self.spk_rec.append(spk_out)
            self.mem_rec.append(mem_out)

        return torch.stack(self.spk_rec), torch.stack(self.mem_rec)

def train_snn(net, train_loader, test_loader, loss, num_epochs, optimizer, num_steps,
               forward=True, SF_funct=False,  infer_loader=None, in2spk=False, device="cpu"):
    """
    A general training function for Spiking Neural Networks (SNN).

    Args:
        net (nn.Module): The SNN model to be trained.
        train_loader (DataLoader): DataLoader for the training dataset.
        test_loader (DataLoader): DataLoader for the test dataset.
        loss (function): Loss function used for training (e.g., CrossEntropyLoss).
        num_epochs (int): Number of training epochs.
        optimizer (Optimizer or list): Optimizer(s) for updating network parameters.
        num_steps (int): Number of timesteps for SNN simulation.
        forward (bool): Whether to use a forward pass (useful when network lacks time iteration).
        SF_funct (bool): Whether to apply spike-function-based loss computation.
        infer_loader (DataLoader, optional): DataLoader for an additional inference dataset.
        in2spk (bool): Whether to encode input as spikes (e.g., using rate encoding).
        device (str): The device to run the training on (e.g., "cpu" or "cuda").

    Returns:
        list: Training loss, training accuracy, test accuracy, and inference accuracy lists.
    """

    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0

    train_l_list=[]
    train_acc_list=[]
    infer_acc_list=[]
    test_acc_list=[]

    for epoch in range(num_epochs):
        for X, y in train_loader: 
        # X: feature tensor, shape [batch_size, 1*28*28]. y: label tensor, shape [batch_size].
        # 1 batch size of dataset for one time
            X=X.to(device)
            y=y.to(device)

            net.train()

            # print(X.shape)
            if in2spk is True and forward is True:
                spk_in=rate_encoding_images(X,num_steps)
                # spk_in.to(device)
                # explicit rate encoding
                # if forward is True:
                # When net contains no time step iteraion, especially using nn.Sequential and etc. pack.
                spk_rec, mem_rec=forward_pass(net,num_steps,spk_in).spkin()
                # else:
                #     spk_rec, mem_rec=net(spk_in)
            elif in2spk is True and forward is False:
                raise ValueError("Wrong settings of in2spk and forward!")
            
            elif forward is True:
                spk_rec, mem_rec=forward_pass(net,num_steps,X.view(X.shape[0], -1)).valin()
            else:
                # X=X.view(X.shape[0], -1)
                # print(X.shape)
                spk_rec, mem_rec=net(X.view(X.shape[0], -1))
            
            spk_rec=spk_rec.to(device)
            mem_rec=mem_rec.to(device)

            if isinstance(optimizer,list):
                for opt in optimizer:
                    # print(opt)
                    opt.zero_grad()
            else:
                optimizer.zero_grad()

            _, y_hat_id=spk_rec.sum(dim=0).max(1) # Output: Rate coding
            
            # initialize the total loss value
            loss_val = torch.zeros((1), dtype=torch.float, device=device)

            if SF_funct is True:
                loss_val=loss(spk_rec, y)
            else:
                for step in range(num_steps):
                    loss_val += loss(mem_rec[step], y)      # cross entrophy.
                # loss_val = loss(y_hat, y).sum().to(device)    # .sum(), input is a list or array
            
            loss_val.backward()
            
            if isinstance(optimizer,list):
                for opt in optimizer:
                    opt.step()
            else:
                optimizer.step()
            # opt_i.step()    # update params
            train_l_sum += loss_val.item()
            train_acc_sum += (y_hat_id == y).float().sum().item() 
            n += y.shape[0]
        test_acc = eval_acc(test_loader, net, num_steps, device=device,in2spk=in2spk,forward=forward)
        train_l_list.append(train_l_sum / n)
        train_acc_list.append(train_acc_sum / n)
        test_acc_list.append(test_acc)

        # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f \n'
        #         % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

        if infer_loader is not None:
            infer_acc = eval_acc(infer_loader, net, num_steps, device=device,in2spk=in2spk,forward=forward)
            infer_acc_list.append(infer_acc)
        

    return [train_l_list, train_acc_list, test_acc_list, infer_acc_list]

def eval_acc(data_iter, net, num_steps, device="cpu",in2spk=False,forward=False):
    with torch.no_grad():
        acc_sum, n = 0.0, 0
        # if isinstance(net, torch.nn.Module):
        #     # for _, layer in net.named_modules():
        #     #     if isinstance(layer, DropoutLayer):
        #     #         layer.training=False
        net.eval()

        for X, y in data_iter:
            X=X.to(device)
            y=y.to(device)
            
            if in2spk is True and forward is True:
                spk_in=rate_encoding_images(X,num_steps)
                spk_rec, _=forward_pass(net,num_steps,spk_in).spkin()

            elif in2spk is True and forward is False:
                raise ValueError("Wrong settings of in2spk and forward!")
            
            elif forward is True:
                spk_rec, _=forward_pass(net,num_steps,X.view(X.shape[0], -1)).valin()
            else:
                spk_rec, _=net(X.view(X.shape[0], -1))
            
            _, y_hat_id=spk_rec.sum(dim=0).max(1)
            
            acc_sum += (y_hat_id == y).float().sum().item()
            n += y.shape[0]

        # if isinstance(net, torch.nn.Module):
        #     # for _, layer in net.named_modules():
        #     #     if isinstance(layer, DropoutLayer):
        #     #         layer.training=True
        net.train()
        

        return acc_sum / n
    
def sav_lin_net_paras(net, fpath):
    """
    Save network's parameters

    Using pandas.DataFrame, the parameters can be saved as .csv file.
    """

    lin=0
    lea=0
    layer_dict={}
    for _, layer in net.named_modules():
        if isinstance(layer, nn.Linear):
            layer_dict.update({f"Linear{lin}_weight": layer.weight})
            layer_dict.update({f"Linear{lin}_bias": layer.bias})
            lin+=1

        if isinstance(layer, snn.leaky):
            layer_dict.update({f"Leaky{lea}_beta": layer.beta})
            layer_dict.update({f"Linear{lea}_reset": layer.reset_mechanism})
            lea+=1
    
    df = pd.DataFrame(layer_dict)

    if os.path.exists(fpath):
        u_input=input(f"Warning! File {fpath} exists! Whether to overwrite? (y/n)").strip().lower()
        if u_input == 'y':
            df.to_csv(fpath, index=False)
            print(f"Parameters are saved in {fpath}.")
        if u_input == 'n':
            new_fp=fpath.replace(".csv",f"_new.csv")
            df.to_csv(new_fp, index=False)
            print(f"Parameters are saved in {new_fp}.")
    else:
        df.to_csv(fpath, index=False)
        print(f"Parameters are saved in {fpath}.")

    
    
    

