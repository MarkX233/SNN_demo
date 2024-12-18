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

        spkin: The input is already coded spikes and length is defined as num_steps.

        valin: The input is value.

        evei: The input is event-based frame, the length is uncertain.
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
    
    def evein(self):
        for step in range(self.data.size(0)):
            spk_out, mem_out = self.net(self.data[step])
            self.spk_rec.append(spk_out)
            self.mem_rec.append(mem_out)

        return torch.stack(self.spk_rec), torch.stack(self.mem_rec)

def train_snn(net, train_loader, test_loader, loss, num_epochs, optimizer, num_steps,
               forward=True, eve_in=False, SF_funct=False,  infer_loader=None, in2spk=False, device="cpu"):
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
        eve_in (bool): Whether the input is an event-based frame.
        SF_funct (bool): Whether to apply a loss function from snntorch.functional (e.g., SF.mse_count_loss).
        infer_loader (DataLoader, optional): DataLoader for an additional inference dataset.
            If is None, the results of inference accuracy will return a list of 0.
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
            
            spk_rec,mem_rec=net_run(net,X,num_steps,in2spk,forward,eve_in)
            
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

            del loss_val, X, y, spk_rec, mem_rec

        test_acc = eval_acc(test_loader, net, num_steps, device=device,in2spk=in2spk,forward=forward,eve_in=eve_in)
        train_l_list.append(train_l_sum / n)
        train_acc_list.append(train_acc_sum / n)
        test_acc_list.append(test_acc)

        # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f \n'
        #         % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

        if infer_loader is not None:
            infer_acc = eval_acc(infer_loader, net, num_steps, device=device,in2spk=in2spk,forward=forward,eve_in=eve_in)
            infer_acc_list.append(infer_acc)
        else:
            infer_acc=0
            infer_acc_list.append(infer_acc)    # For plot

        if device == "cuda":
            torch.cuda.empty_cache()
         
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return [train_l_list, train_acc_list, test_acc_list, infer_acc_list]

def eval_acc(data_iter, net, num_steps, device="cpu",in2spk=False,forward=False,eve_in=False):
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

            spk_rec,_=net_run(net,X,num_steps,in2spk,forward,eve_in)
            
            _, y_hat_id=spk_rec.sum(dim=0).max(1)
            
            acc_sum += (y_hat_id == y).float().sum().item()
            n += y.shape[0]

        # if isinstance(net, torch.nn.Module):
        #     # for _, layer in net.named_modules():
        #     #     if isinstance(layer, DropoutLayer):
        #     #         layer.training=True
        net.train()
        

        return acc_sum / n
    
def sav_lin_net_paras(net, path, dname, index=False, optimizer=None):
    """
    Save network's parameters

    Using pandas.DataFrame, the parameters can be saved as .csv file.
    """

    lin=0
    lea=0
    layer_dict={}

    dirpath=os.path.join(path, dname)

    if os.path.exists(dirpath):
        u_input=input(f"Warning! Directory {dirpath} exists! Whether to overwrite? (y/n)").strip().lower()
        if u_input == 'y':
            for filename in os.listdir(dirpath):
                file_path = os.path.join(dirpath, filename)
                try:
                    if os.path.isfile(file_path):
                        os.remove(file_path)
                        print(f"Deleted file: {file_path}")
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")
            print(f"Parameters will be saved in {dirpath}.")
        if u_input == 'n':
            dirpath=dirpath+"_new"
            os.makedirs(dirpath)
            print(f"Parameters will be saved in {dirpath}.")
    else:
        os.makedirs(dirpath)

    for _, layer in net.named_modules():
        if isinstance(layer, nn.Linear):
            layer_dict[f"Linear{lin}_weight"] = layer.weight.flatten().detach().cpu().tolist()
            layer_dict[f"Linear{lin}_bias"] = layer.bias.flatten().detach().cpu().tolist()
            lin += 1

        if isinstance(layer, snn.Leaky):
            layer_dict[f"Leaky{lea}"] = {
                "Beta":layer.beta.detach().cpu().tolist(),
                "Reset mechanism":layer.reset_mechanism_val.detach().cpu().tolist()                         
                }
            lea += 1

    for key, value in layer_dict.items():
        if isinstance(value, dict):
            df = pd.DataFrame.from_dict(value,orient='index')
        else:
            df = pd.DataFrame(value)
        fpath=os.path.join(dirpath, f"{key}.csv")
        df.to_csv(fpath, index=index)
        print(f"Parameters are saved in {fpath}.")

    if optimizer is not None:
        torch.save(optimizer.state_dict(), f"{dirpath}/optimizer_state.pth")
        print(f"Optimizer state is saved in {dirpath}/optimizer_state.pth")

    torch.save(net.state_dict(), f"{dirpath}/net_params.pth")
    print(f"Network state is saved in {dirpath}/net_params.pth")

    print("All parameters are saved.")
    
# def load_net_params(net,dir):

#     lin = 0
#     lea = 0
#     weight_tensor=[]
#     bias_tensor=[]

#     if not os.path.exists(dir):
#         raise FileNotFoundError(f"Directory {dir} does not exist. Check the path and directory name!")
    
#     # Load parameters into memory
#     for filename in sorted(os.listdir(dir)):
#         fpath = os.path.join(dir, filename)

#         # Skip if not a .csv file
#         if not filename.endswith(".csv"):
#             continue

#         # Load the parameter data
#         df = pd.read_csv(fpath, header=None)

#         # Load Linear layers
#         if f"Linear{lin}_weight" in filename:
#             weight_tensor.append(df[0].tolist())
#         if f"Linear{lin}_bias" in filename:
#             bias_tensor.append(df[0].tolist())
        
            
#     i=0
#     for _, layer in net.named_modules():
#         with torch.no_grad():
#             if isinstance(layer, nn.Linear):
#                 if lin==i:
#                     if len(layer.weight)==len(weight_tensor):
#                         layer.weight.data=weight_tensor
#                         print(f"Loaded Linear{lin} weights from {filename}.")
#                     else:
#                         raise ValueError("Error! The length of saved linear weights is different from the settings of the net.")
#                     if len(layer.bias)==len(bias_tensor):
#                         layer.bias.data=bias_tensor
#                         print(f"Loaded Linear{lin} bias from {filename}.")
#                     else:
#                         raise ValueError("Error! The length of saved linear bias is different from the settings of the net.")
#                     lin += 1
#             if isinstance(layer, snn.Leaky):
#                 if lea==i:
#                     pass
#                     #


#         # # Load Leaky layers
#         # elif f"Leaky{lea}" in filename:
#         #     leaky_param = df.values.flatten()
#         #     if "Beta" in filename:
#         #         net[lea].beta.data = torch.tensor(leaky_param)
#         #         print(f"Loaded Leaky{lea} beta from {filename}.")
#         #     elif "Reset mechanism" in filename:
#         #         net[lea].reset_mechanism_val.data = torch.tensor(leaky_param)
#         #         print(f"Loaded Leaky{lea} reset mechanism from {filename}.")
#         #         lea += 1

#     print("All parameters have been loaded successfully.")
#     return net
    
    
def net_run(net,X,num_steps,in2spk,forward,eve_in):
    if in2spk is True and eve_in is True:
        raise ValueError("Error! Wrong settings of in2spk and eve_in! These two can't stay True together!")
            
    if in2spk is True and forward is False:
        raise ValueError("Error! Wrong settings of in2spk and forward! When `in2spk` is set to True, `forward` must also be set to True.")
    
    if eve_in is True and forward is False:
        raise ValueError("Error! Wrong settings of eve_in and forward! When `eve_in` is set to True, `forward` must also be set to True.")


    if in2spk is True and forward is True:
        spk_in=rate_encoding_images(X,num_steps)
        # spk_in.to(device)
        # explicit rate encoding
        # if forward is True:
        # When net contains no time step iteraion, especially using nn.Sequential and etc. pack.
        spk_rec, mem_rec=forward_pass(net,num_steps,spk_in).spkin()
        # else:
        #     spk_rec, mem_rec=net(spk_in)
    elif eve_in is True and forward is True:
        spk_rec, mem_rec=forward_pass(net,num_steps,X).evein()
    
    elif forward is True:
        spk_rec, mem_rec=forward_pass(net,num_steps,X.view(X.shape[0], -1)).valin()
    else:
        # X=X.view(X.shape[0], -1)
        # print(X.shape)
        spk_rec, mem_rec=net(X.view(X.shape[0], -1))

    return spk_rec, mem_rec