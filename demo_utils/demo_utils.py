from torch import nn
import torch
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
from collections import OrderedDict
from pathvalidate import sanitize_filename
from torch.nn import init
import pandas as pd
# from collections import OrderedDict
import os
import re



class FlattenLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x): # self-made layer, forward arrguments only contain x
        return x.view(x.shape[0], -1)
    
class ReLuFunction(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x): 
        return torch.max(input=x, other=torch.tensor(0.0))
    
class DropoutLayer(nn.Module):
    def __init__(self, drop_prob, training=True):
        super().__init__()
        self.drop_prob=drop_prob
        self.training=training
    def forward(self, x): 
        if self.training==True:
            assert 0 <= self.drop_prob <= 1
            keep_prob = 1 - self.drop_prob
            mask = (torch.rand(x.shape) < keep_prob).float()
            return mask * x / keep_prob
        else:
            return x

def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition  

def cross_entropy(y_hat, y):
    return - torch.log(y_hat.gather(1, y.view(-1, 1)))

def cross_entropy_2(y_hat, y):
    l=torch.tensor([])
    for row in range(len(y_hat)):
        # print(y_hat[row, y[row]])
        l=torch.cat((l,-torch.log(y_hat[row, y[row]]).unsqueeze(0)))
    return l

def sgd(params, lr, batch_size):
    for param in params:
        # print(param.grad)
        # print(type(param.grad))
        param.data -= lr * param.grad / batch_size

def evaluate_accuracy(data_iter, net, params=None, device="cpu"):  # calculate accuracy without updating weights
    acc_sum, n = 0.0, 0
    if isinstance(net, torch.nn.Module):
        for _, layer in net.named_modules():
            if isinstance(layer, DropoutLayer):
                layer.training=False
        net.eval()

    for X, y in data_iter:
        X=X.to(device)
        y=y.to(device)
        if params is not None:
            acc_sum += (net(X,params=params).argmax(dim=1) == y).float().sum().item()
        else:
            acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]

    if isinstance(net, torch.nn.Module):
        for _, layer in net.named_modules():
            if isinstance(layer, DropoutLayer):
                layer.training=True
        net.train()
    

    return acc_sum / n


def train_funct(net, train_iter, test_iter, loss, num_epochs, batch_size, optimizer,
              params=None, lr=None, selfbuild=False, dc_lambda=None,  infer_iter=None,
              device="cpu"):
    
    train_l_sum, train_acc_sum, n = 0.0, 0.0, 0

    train_l_list=[]
    train_acc_list=[]
    infer_acc_list=[]
    test_acc_list=[]

    for epoch in range(num_epochs):
        for X, y in train_iter: 
        # X: feature tensor, shape [batch_size, 1*28*28]. y: label tensor, shape [batch_size].
        # 1 batch size of dataset for one time
            X=X.to(device)
            y=y.to(device)
            if selfbuild==True:
                y_hat = net(X,params=params)
            else:
                y_hat = net(X)
                if isinstance(optimizer,list):
                    for opt in optimizer:
                        # print(opt)
                        opt.zero_grad()
                else:
                    optimizer.zero_grad()
            l = loss(y_hat, y).sum().to(device)    # .sum(), input is a list or array
            if dc_lambda is not None:
                for _, layer in net.named_modules():    
                    if isinstance(layer, nn.Linear):
                        l2=dc_lambda * layer.weight **2 / 2
                        l+=l2.sum()

            if params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()
            l.backward()
            if selfbuild==True: 
                optimizer(params=params, lr=lr, batch_size=batch_size)
            else:
                if isinstance(optimizer,list):
                    for opt in optimizer:
                        opt.step()
                else:
                    optimizer.step()
            # opt_i.step()    # update params
            train_l_sum += l.item()
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item() # argmax returns index
            n += y.shape[0]
        test_acc = evaluate_accuracy(test_iter, net, params=params, device=device)
        train_l_list.append(train_l_sum / n)
        train_acc_list.append(train_acc_sum / n)
        test_acc_list.append(test_acc)

        # print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f \n'
        #         % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))

        if infer_iter is not None:
            infer_acc = evaluate_accuracy(infer_iter, net, params=params, device=device)
            infer_acc_list.append(infer_acc)
        

    return [train_l_list, train_acc_list, test_acc_list, infer_acc_list]

def plot_acc(epoch, train_l_list, train_acc_list, test_acc_list, infer_acc_list, suptitle=None, sub_title=None, xlist=None, store_pic=True):
    
    if xlist is not None:
        epoch_list=xlist    # Using other variables list
    else:
        epoch_list = list(range(1, epoch + 1))

    fig=plt.figure(figsize=(10, 7))

    plt.subplot(2,2,1)
    # plt.tight_layout()
    plot_xy(epoch_list, train_l_list, title="Loss")
    plt.text(epoch_list[-1], train_l_list[-1], f'{train_l_list[-1]:.6f}', fontsize=8, ha='left', va='bottom')

    plt.subplot(2,2,2)
    plot_xy(epoch_list, train_acc_list, title="Train Accuraccy")
    plt.text(epoch_list[-1], train_acc_list[-1], f'{train_acc_list[-1]:.6f}', fontsize=8, ha='left', va='bottom')
    
    plt.subplot(2,2,3)
    plot_xy(epoch_list, test_acc_list, title="Test Accuraccy")
    plt.text(epoch_list[-1], test_acc_list[-1], f'{test_acc_list[-1]:.6f}', fontsize=8, ha='left', va='bottom')

    try:
        plt.subplot(2,2,4)
        plot_xy(epoch_list, infer_acc_list, title="Inference Accuraccy")
        plt.text(epoch_list[-1], infer_acc_list[-1], f'{infer_acc_list[-1]:.6f}', fontsize=8, ha='left', va='bottom')
    except ValueError:
        print("No inference data!")
    else:
        pass

    plt.subplots_adjust(hspace=0.4,wspace=0.4)
    plt.suptitle(suptitle)
    fig.text(0.5,0.925,sub_title,fontsize=8,color="blue",ha="center", va="center")
    if store_pic is True:
        os. makedirs("rec",exist_ok=True)
        plt.savefig(f"rec/{sanitize_filename(suptitle)}.svg")
    plt.show()

def plot_xy(x, y, xlabel=None, ylabel=None, title=None, save=None, figsize=(10, 5),show=False):
    set_figsize(figsize)
    plt.plot(x, y)
    plt.xlabel(xlabel, fontsize=10)
    plt.ylabel(ylabel, fontsize=10)
    plt.title(title, fontsize=12)
    if show is True:
        plt.show()
    if save is not None:
        os. makedirs("pic",exist_ok=True)
        plt.savefig(f"pic/{sanitize_filename(save)}.svg")

def use_svg_display():
    """Use svg format to display plot in jupyter"""
    display.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def build_network(num_hidden_layers, num_hiddens_ord, act_ord, num_inputs, num_outputs, dp_layer=[], dp_prob_ord=[]):
    """
    Builds a neural network with the specified structure and parameters.

    Args:
        num_hidden_layers (int): 
            Number of hidden layers in the network. If `num_hidden_layers` exceeds the 
            number of neurons or activation functions defined, excess values will be ignored.
        num_hiddens_ord (list of int): 
            Number of neurons in each hidden layer, in order. Must have at least 
            `num_hidden_layers` elements; otherwise, an error will be raised.
        act_ord (list): 
            Activation functions for the hidden layers. Each value corresponds to a layer, 
            defined by the `add_act_funct` function. If not enough values are provided, 
            subsequent layers will use the default activation function.
            Possible values include: `0`, `"re"`, `"sig"`, `"tanh"`, `"lkre"`, `"elu"`.
        num_inputs (int): 
            Number of input features for the first layer.
        num_outputs (int): 
            Number of output features for the final layer.
        dp_layer (list of int, optional): 
            Indices of hidden layers after which dropout should be applied. Starts from 0, 
            where 0 refers to the layer after the input-to-first-hidden connection. Defaults to an empty list.
        dp_prob_ord (list of float, optional): 
            Dropout probabilities, corresponding to `dp_layer` order. For example:
            `dp_layer=[1,2]` and `dp_prob_ord=[0.1, 0.2]` adds dropout with probabilities 0.1 
            and 0.2 after the 1st and 2nd hidden layers, respectively. Defaults to an empty list.

    Returns:
        OrderedDict: 
            A sequentially ordered dictionary of layers, which can be used to construct 
            the network with `nn.Sequential()`.

    Notes:
        - The function ensures the number of activation functions and dropout probabilities 
          match the number of layers. Warnings are printed if insufficient values are provided.
        - Use `nn.Sequential(build_network(...)).to(device)` to construct the network 
          and move it to the desired device.

    Raises:
        ValueError: 
            If `num_hidden_layers` exceeds the length of `num_hiddens_ord`.

    Example:
        network = nn.Sequential(build_network(
            num_hidden_layers=3, 
            num_hiddens_ord=[128, 64, 32], 
            act_ord=["re", "tanh", "elu"], 
            num_inputs=784, 
            num_outputs=10, 
            dp_layer=[0, 2], 
            dp_prob_ord=[0.2, 0.5]
        )).to(device)
    """

    noact=False

    if (num_hidden_layers > len(num_hiddens_ord)):
        raise ValueError("Error! Nummber of hidden layers is larger than the hidden layers that's been defined!")
    
    elif len(act_ord)==0 or act_ord is None:
        noact=True
        print("Warning! No defined active funtion is Set. \n \
              The layers will use default active funtion")

    elif num_hidden_layers > len(act_ord):
        print("Warning! Defined active funtions are not enough for the defined layers. \n \
              The rest of layers will use default active funtion")

    layers = [('flatten', nn.Flatten())]

    layers.append((f'linear_in', nn.Linear(num_inputs, num_hiddens_ord[0])))

    if noact is True:
        add_act_funct(layers, act_funct="re", i=0)
    else:
        add_act_funct(layers, act_funct=act_ord[0], i=0)

    j=0
    nodrop=False

    try: 
        if dp_layer[0]==0:
            layers.append((f'dropout0', nn.Dropout(dp_prob_ord[0])))
            j+=1
    except IndexError:
        nodrop=True
        print("Warning! No drop out layer is set!")

    
    # print (act_ord)
    # print (len(act_ord))
    
    for i in range(1,num_hidden_layers):
        layers.append((f'linear{i}', nn.Linear(num_hiddens_ord[i-1], num_hiddens_ord[i])))
        if noact is True:
            add_act_funct(layers, act_funct="re", i=i)
        elif i < len(act_ord):
            add_act_funct(layers, act_ord[i], i)  
        else:
            add_act_funct(layers, act_funct="re", i=i)
        
        if j>=len(dp_layer):
            pass
        elif nodrop is False and i==dp_layer[j]:
            layers.append((f'dropout{i}', nn.Dropout(dp_prob_ord[j])))
            j+=1

    layers.append((f'linear_out', nn.Linear(num_hiddens_ord[num_hidden_layers-1], num_outputs)))

    return OrderedDict(layers)

def add_act_funct(layers, act_funct="re", i=0):
    match act_funct:
        case "0":    # add no active function
            return
        case "re":
            layers.append((f'active{i}', nn.ReLU()))
        case "sig":
            layers.append((f'active{i}', nn.Sigmoid()))
        case "tanh":
            layers.append((f'active{i}', nn.Tanh()))
        case "lkre":
            layers.append((f'active{i}', nn.LeakyReLU(negative_slope=0.01)))
        case "elu":
            layers.append((f'active{i}', nn.ELU(alpha=1.0)))

def build_optimizer(net, lr_ord,opt_type_ord=[], decay_layer=[], decay_rate=[]):
    """
    Builds optimizers for the given neural network layers with optional weight and bias decay.

    Args:
        net (torch.nn.Module): 
            The neural network model.
        lr_ord (float or list): 
            Learning rate(s). If a single float is provided, it is used for all layers. 
            If a list is provided, each value corresponds to the learning rate for each layer.
        opt_type_ord (str or list, optional): 
            Optimizer type(s). If a single string is provided, the same optimizer type is 
            applied to all layers. If a list is provided, each value corresponds to the optimizer 
            type for each layer. Defaults to "sgd" if not specified.
        decay_layer (list, optional): 
            List of layer indices where weight and bias decay should be applied. 
            Layer indices start at 0, where 0 refers to the input-to-hidden1 connection. 
            Defaults to an empty list (no decay).
        decay_rate (list, optional): 
            List of decay rates corresponding to `decay_layer`. Each element is a list of 
            length 2, where the first value is the weight decay rate, and the second value is the bias decay rate.
            Example: `decay_rate=[[0.1, 0.2]]` applies a weight decay rate of 0.1 and a bias decay rate of 0.2 
            to the first layer (layer index in `decay_layer[0]`). Defaults to an empty list.

    Returns:
        list: 
            A list of optimizers for the network layers. Includes separate optimizers for 
            weights and biases if decay is applied.

    Raises:
        ValueError: 
            If the decay rate format is invalid, or the number of `decay_layer` elements 
            does not match the number of `decay_rate` elements.
        ValueError: 
            If `net` is not an instance of `torch.nn.Module`.

    Notes:
        - If `decay_layer` and `decay_rate` are not provided, no decay will be applied.
        - The function ensures that learning rates and optimizer types are matched to the layers.
        - By default, "sgd" is used as the optimizer type if `opt_type_ord` is not specified.

    Example:
        optimizer = build_optimizer(
            net=model,
            lr_ord=[0.01, 0.001],
            opt_type_ord=["sgd", "adam"],
            decay_layer=[1],
            decay_rate=[[0.1, 0.2]]
        )
    """
    optimizer=[]
    opt_w=[]
    opt_b=[]

    if len(decay_layer)==0 or len(decay_rate)==0 or decay_rate is None or decay_layer is None:
        nodecay=True
        print("Warning! No decay layer or decay rate is set! Using no decay set!")
    else:
        nodecay=False
    
    if nodecay is True:
        pass
    elif validate_decay_rate(decay_rate)==False:
        raise ValueError("Error! Unlegal input of decay rate!")        
    elif len(decay_layer)!=len(decay_rate):
        raise ValueError("Error! Set of decay_layer doesn't meet the set of decay rate.")
    
    
    
    if isinstance(net, torch.nn.Module):
        num_lin_layer=0
        j=0
        for _, layer in net.named_modules():

            if isinstance(lr_ord,list)==False:
                lr_t=lr_ord
            else:
                lr_t=lr_ord[num_lin_layer]
            
            if len(opt_type_ord)==0:
                opt_type_t="sgd"
                print("Warning! No optimizer type is set! Using default: sgd.")
            elif isinstance(opt_type_ord,list)==False:
                opt_type_t=opt_type_ord
            else:
                opt_type_t=opt_type_ord[num_lin_layer]

            if isinstance(layer, nn.Linear):
                match opt_type_t:
                    case "sgd":
                        if nodecay is False and j<len(decay_layer) and num_lin_layer==decay_layer[j]:
                            opt_w.append(torch.optim.SGD([layer.weight], lr=lr_t, weight_decay=decay_rate[j][0]))
                            opt_b.append(torch.optim.SGD([layer.bias], lr=lr_t, weight_decay=decay_rate[j][1])) 
                            j+=1
                        else:
                            optimizer.append(torch.optim.SGD(layer.parameters(), lr=lr_t))
                    case "adam":
                        if nodecay is False and j<len(decay_layer) and num_lin_layer==decay_layer[j]:
                            opt_w.append(torch.optim.Adam([layer.weight], lr=lr_t, weight_decay=decay_rate[j][0]))
                            opt_b.append(torch.optim.Adam([layer.bias], lr=lr_t, weight_decay=decay_rate[j][1])) 
                            j+=1
                        else:
                            optimizer.append(torch.optim.Adam(layer.parameters(), lr=lr_t))
                    case "adaw":
                        if nodecay is False and j<len(decay_layer) and num_lin_layer==decay_layer[j]:
                            opt_w.append(torch.optim.AdamW([layer.weight], lr=lr_t, weight_decay=decay_rate[j][0]))
                            opt_b.append(torch.optim.AdamW([layer.bias], lr=lr_t, weight_decay=decay_rate[j][1])) 
                            j+=1
                        else:
                            optimizer.append(torch.optim.AdamW(layer.parameters(), lr=lr_t))
                    case "rms":
                        if nodecay is False and j<len(decay_layer) and num_lin_layer==decay_layer[j]:
                            opt_w.append(torch.optim.RMSprop([layer.weight], lr=lr_t, weight_decay=decay_rate[j][0]))
                            opt_b.append(torch.optim.RMSprop([layer.bias], lr=lr_t, weight_decay=decay_rate[j][1])) 
                            j+=1
                        else:
                            optimizer.append(torch.optim.RMSprop(layer.parameters(), lr=lr_t))
                    case "adag":
                        if nodecay is False and j<len(decay_layer) and num_lin_layer==decay_layer[j]:
                            opt_w.append(torch.optim.Adagrad([layer.weight], lr=lr_t, weight_decay=decay_rate[j][0]))
                            opt_b.append(torch.optim.Adagrad([layer.bias], lr=lr_t, weight_decay=decay_rate[j][1])) 
                            j+=1
                        else:
                            optimizer.append(torch.optim.Adagrad(layer.parameters(), lr=lr_t))
                    case "adad":
                        if nodecay is False and j<len(decay_layer) and num_lin_layer==decay_layer[j]:
                            opt_w.append(torch.optim.Adadelta([layer.weight], lr=lr_t, weight_decay=decay_rate[j][0]))
                            opt_b.append(torch.optim.Adadelta([layer.bias], lr=lr_t, weight_decay=decay_rate[j][1])) 
                            j+=1
                        else:
                            optimizer.append(torch.optim.Adadelta(layer.parameters(), lr=lr_t))
                    case "adamx":
                        if nodecay is False and j<len(decay_layer) and num_lin_layer==decay_layer[j]:
                            opt_w.append(torch.optim.Adamax([layer.weight], lr=lr_t, weight_decay=decay_rate[j][0]))
                            opt_b.append(torch.optim.Adamax([layer.bias], lr=lr_t, weight_decay=decay_rate[j][1])) 
                            j+=1
                        else:
                            optimizer.append(torch.optim.Adamax(layer.parameters(), lr=lr_t))
                    case "asgd":
                        if nodecay is False and j<len(decay_layer) and num_lin_layer==decay_layer[j]:
                            opt_w.append(torch.optim.ASGD([layer.weight], lr=lr_t, weight_decay=decay_rate[j][0]))
                            opt_b.append(torch.optim.ASGD([layer.bias], lr=lr_t, weight_decay=decay_rate[j][1])) 
                            j+=1
                        else:
                            optimizer.append(torch.optim.ASGD(layer.parameters(), lr=lr_t))
                    case "nada":
                        if nodecay is False and j<len(decay_layer) and num_lin_layer==decay_layer[j]:
                            opt_w.append(torch.optim.NAdam([layer.weight], lr=lr_t, weight_decay=decay_rate[j][0]))
                            opt_b.append(torch.optim.NAdam([layer.bias], lr=lr_t, weight_decay=decay_rate[j][1])) 
                            j+=1
                        else:
                            optimizer.append(torch.optim.NAdam(layer.parameters(), lr=lr_t))

                            
                    case _:
                        if nodecay is False and j<len(decay_layer) and num_lin_layer==decay_layer[j]:
                            opt_w.append(torch.optim.SGD([layer.weight], lr=lr_t, weight_decay=decay_rate[j][0]))
                            opt_b.append(torch.optim.SGD([layer.bias], lr=lr_t, weight_decay=decay_rate[j][1]))
                            j+=1
                        else:
                            optimizer.append(torch.optim.SGD(layer.parameters(), lr=lr_t))
        
                num_lin_layer+=1
    else: raise ValueError("Error! Wrong net module type! Please use torch.nn.Module.") 
    
    optimizer=optimizer+opt_w+opt_b

    return optimizer


def validate_decay_rate(input_list):
    """
    Verify that the input is a list which meets the requirements.
    Requirement: The input is a list, and each element in the list is also a list of length 2.
    """
    if input_list is None:
        return True         # decay_rate is allowed to be None

    elif not isinstance(input_list, list):
        print("Input must be a list.")
        return False
    elif len(input_list)==0:
        return True
    
    for index, item in enumerate(input_list):
        if not isinstance(item, list):
            print(f"The input [{index}] ({item}) is not a list!")
            return False
        if len(item) != 2:
            print(f"The length of input list [{index}] ({item}) is not 2.")
            return False
        if not all(isinstance(sub_item, (int, float)) for sub_item in item):
            print(f"The list [{index}] ({item}) contains non-numeric elements.")
            return False
    
    return True



name_to_abbr = {
    "batch_size": "bs",
    "num_inputs": "ni",
    "num_hiddens": "nh",
    "num_outputs": "no",
    "num_epochs": "ne",
    "lr": "lr",
    "suptitle": "st",
    "num_hidden_layers": "nhl",
    "num_hiddens_ord": "nho",
    "act_ord":"ao",
    "dp_layer":"dpl",
    "dp_prob_ord":"dpo",
    "lr_ord":"lro",
    "opt_type_ord": "oto",
    "decay_layer": "dcl",
    "decay_rate": "dcr",
    "loss_funct": "lf"
}

abbr_to_name = {v: k for k, v in name_to_abbr.items()}


def encode_params2hpid(params):
    """
    Encode the parameter dictionary as a string ID
    """
    id_parts = []
    for key, value in params.items():
        if key in name_to_abbr:
            abbr = name_to_abbr[key]
            id_parts.append(f"{abbr}={value}")
    return "-".join(id_parts)

import json

def decode_hpid(encoded_id):
    """
    Decode the string ID back into a parameter dictionary
    """
    decoded_params = {}
    parts = encoded_id.split("-")
    for part in parts:
        abbr, value = part.split("=")
        if abbr in abbr_to_name:
            key = abbr_to_name[abbr]
            try:
                if value.startswith("[") and value.endswith("]"):
                    try:
                        value = json.loads(value)
                    except json.JSONDecodeError:
                        value = eval(value)
                elif "." in value:
                    value = float(value)
                else:
                    value = int(value)
            except ValueError:
                pass
            decoded_params[key] = value
        else:
            print(f"Warning! HPID includes unrecognized part: -{abbr}")
    return decoded_params

def reproduce(hpid, train_dataset, test_dataset,infer_dataset,remark=[],debug_mode=False):

    num_workers=4

    hparams=decode_hpid(hpid)

    if debug_mode:
        print(hparams)

    train_iter = torch.utils.data.DataLoader(train_dataset, batch_size=hparams["batch_size"], shuffle=True, num_workers=num_workers)
    test_iter = torch.utils.data.DataLoader(test_dataset, batch_size=hparams["batch_size"], shuffle=False, num_workers=num_workers)
    infer_iter = torch.utils.data.DataLoader(infer_dataset, batch_size=hparams["batch_size"], shuffle=False, num_workers=num_workers)

    results=imp_nn_and_train(hparams=hparams,train_iter=train_iter,test_iter=test_iter,infer_iter=infer_iter,
                 remark=remark,debug_mode=debug_mode,x="Reproduce")
    
    return results

    
    

def vary_check(vary_type, hparams):
    """
    ## Check the vary type is whether in hparams
    ## Return bool
    """

    for key, _ in hparams.items():
        if  vary_type==key:
            return True

    return False

def get_loss_function(loss_name, **kwargs):
    """
    Select a loss function based on the given name.
    
    Args:
        loss_name (str): The name of the loss function. Must be one of the supported names.
        kwargs (dict): Additional arguments for specific loss functions.
    
    Returns:
        loss_fn (nn.Module): The selected loss function.
    
    Supported Loss Functions:
        - `"mse"`, `"ce"`, `"nll"`, `"bce"`, `"bcel"` , `"sml1"`, `"huber"`, `"kldiv"`, `"hinge"`, `"cosine"`
    """
    loss_dict = {
        "mse": nn.MSELoss,
        "ce": nn.CrossEntropyLoss,
        "nll": nn.NLLLoss,
        "bce": nn.BCELoss,
        "bcel": nn.BCEWithLogitsLoss,
        "sml1": nn.SmoothL1Loss,
        "huber": nn.HuberLoss,
        "kldiv": nn.KLDivLoss,
        "hinge": nn.HingeEmbeddingLoss,
        "cosine": nn.CosineEmbeddingLoss
    }

    if loss_name is None or len(loss_name)==0:
        return loss_dict["ce"](**kwargs)
    # Default
    
    elif loss_name not in loss_dict:
        raise ValueError(f"Unsupported loss function '{loss_name}'. Supported options are: {list(loss_dict.keys())}")
    
    return loss_dict[loss_name](**kwargs)


def vt2suptitle(vary_type):
    match vary_type:
        case "batch_size":
            return "Batch Size"
        case "num_inputs":
            return "Number of Inputs"
        case "num_outputs":
            return "Number of Outputs"
        case "num_epochs":
            return "Number of Epochs"
        case "suptitle":
            return "Superscript Title"
        case "num_hidden_layers":
            return "Number of Hidden Layers"
        case "num_hiddens_ord":
            return "Hidden Layers' Neuron Numbers"
        case "act_ord":
            return "Activation Functions "
        case "dp_layer":
            return "Dropout Layers"
        case "dp_prob_ord":
            return "Dropout Probabilities "
        case "lr_ord":
            return "Learning Rate "
        case "opt_type_ord":
            return "Optimizer Type "
        case "decay_layer":
            return "Decay Layers"
        case "decay_rate":
            return "Decay Rates (Per Layer)"
        case "loss_funct":
            return "Loss Function"
        case _:
            print(f"Warning! No match key found in input dict for '{vary_type}'!")
            return None

def imp_nn_and_train(hparams,train_iter, test_iter, infer_iter, 
                 remark=[],debug_mode=False,x=None):
    """
    ## Implement neuron network by hparams, perform training and plot results
    ## Return results | dict
    """

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    net = nn.Sequential(build_network(
                    hparams["num_hidden_layers"],
                    hparams["num_hiddens_ord"],
                    hparams["act_ord"],
                    hparams["num_inputs"],
                    hparams["num_outputs"],
                    dp_layer=hparams.get("dp_layer",[]),    # Unnecessary hparams
                    dp_prob_ord=hparams.get("dp_prob_ord",[])
                )).to(device)
        
    if debug_mode is True:
        print(net)

    for _, layer in net.named_modules():
        if isinstance(layer, nn.Linear):
            
            init.normal_(layer.weight, mean=0, std=0.01)
            init.constant_(layer.bias, val=0)

    loss = get_loss_function(hparams["loss_funct"])

    if debug_mode is True:
        print(f"{loss}\n")

    # optimizer = torch.optim.SGD(net.parameters(), hparams["lr"])

    optimizer=build_optimizer(net,hparams["lr_ord"],opt_type_ord=hparams["opt_type_ord"],
                              decay_layer=hparams.get("decay_layer",[]),decay_rate=hparams.get("decay_rate",[]))
    
    if debug_mode is True:
        for opt in optimizer:
            print(f"{opt}\n")

    results=train_funct(net, train_iter, test_iter, loss, hparams["num_epochs"],
                         hparams["batch_size"],optimizer , params=None, infer_iter=infer_iter, device=device)

    print("Training complete!")

    hpid_o=encode_params2hpid(hparams)
    print(f"Encoded HPID: {hpid_o}")
    
    plot_acc(hparams["num_epochs"],results[0],results[1],results[2],results[3],
             f"{hparams["suptitle"]}:{x}_{remark}", hpid_o)

    return results, hpid_o

    
def iter_train(vary_type,vary_list,hparams_i,train_iter, test_iter, infer_iter, 
                 remark=[],debug_mode=False,store_round=False,store_final=False,
                 hpid_i=None):
    
    """
    Iteration training set by vary_type and vary_list
    """

    record={}

    # Use HPID to form a network
    if hpid_i is not None:
        hparams=decode_hpid(hpid_i)
    else:
        hparams=hparams_i

    if vary_check(vary_type,hparams) is False:
        raise ValueError("Error! Wrong type of varriable!")

    for x in vary_list:
        hparams[vary_type]=x    # Update value of hparams from vary list
    

        results, hpid=imp_nn_and_train(hparams,train_iter,test_iter,infer_iter,remark=remark,
                             debug_mode=debug_mode,x=x)

        record[f"{hparams["suptitle"]}: {x}"]={
            "HPID": hpid,
            "Train Loss": results[0],
            "Train Accuracy": results[1],
            "Test Accuracy": results[2],
            "Infer Accuracy": results[3],
        }

        if store_final is True:
        # Store final results for in one file
            store_final_results(results,f"{hparams["suptitle"]}-{remark}",x,hpid)
        
    
    # Store results for this round.
    if store_round is True:
        store_record(record,f"{hparams["suptitle"]}-{remark}")

    print("All iteration training is complete!")

    

def store_record(record,filename, one_time=False):
    """
    Store iteration's record of results.

    Args:
        record (dict | list):
            If `one_time` is false, it is dict of results:

            Example::

                record[f"{hparams["suptitle"]}: {x}"]={
                "HPID": hpid,
                "Train Loss": results[0],
                "Train Accuracy": results[1],
                "Test Accuracy": results[2],
                "Infer Accuracy": results[3],
            
            If `one_time` is true, record is a list of results
        }
        filename (str):
            filename not path.
        one_time (bool):
            If True, only store one result of training
    """
    file_path = f"rec/{filename}.csv"

    flat_record = []
    if one_time is True:
        flat_record={
            "Title": filename,
            "Train Loss": record[0],
            "Train Accuracy": record[1],
            "Test Accuracy": record[2],
            "Infer Accuracy": record[3],
        }
    else:
        for key, value in record.items():
            flat_row = {"Experiment": key}
            flat_row.update(value)
            flat_record.append(flat_row)

    df = pd.DataFrame(flat_record)

    if os.path.exists(file_path):
        existing_df = pd.read_csv(file_path)
        combined_df = pd.concat([existing_df, df])
        if one_time is False:
            combined_df = combined_df.drop_duplicates(subset="HPID", keep="last")
        # Same HPID record will be overwroten.
    else:
        combined_df = df

    if one_time is False:
        combined_df.sort_values(by="Experiment", ascending=True, inplace=True)
    combined_df.to_csv(file_path, index=False)
    print(f"Data of {filename} is saved!")

def store_final_results(results, titlename, vary_x=None, hpid=None, single_file=True, total_file=True):
    """
    Store final results for in one file
    """
    
    final_record={
        "Title": titlename,
        "Varialbe Value": vary_x,
        "HPID": hpid,
        "Train Loss": results[0][-1],
        "Train Accuracy": results[1][-1],
        "Test Accuracy": results[2][-1],
        "Infer Accuracy": results[3][-1],
        }

    tfile_path = f"rec/total_results.csv"

    tdf=pd.DataFrame(final_record,index=[0])

    sinfile_path=f"rec/{titlename}_finals.csv"

    if single_file is True:

        if os.path.exists(sinfile_path):
            try:
                existing_tdf = pd.read_csv(sinfile_path)
                combined_tdf = pd.concat([existing_tdf, tdf])
                combined_tdf = combined_tdf.drop_duplicates(subset="HPID", keep="last")
            except (pd.errors.EmptyDataError, ValueError):
                print(f"Warning: Can't add final result into {titlename}: {vary_x} into {sinfile_path}!")
        else:
            combined_tdf = tdf

        combined_tdf.sort_values(by="Title", ascending=True, inplace=True)
        combined_tdf.to_csv(sinfile_path, index=False)
        print(f"Data of final results of {titlename}: {vary_x} is saved!")

    if total_file is True:

        if os.path.exists(tfile_path):
            try:
                existing_tdf = pd.read_csv(tfile_path)
                combined_tdf = pd.concat([existing_tdf, tdf])
                combined_tdf = combined_tdf.drop_duplicates(subset="HPID", keep="last")
                # Same HPID record will be overwroten.
            except (pd.errors.EmptyDataError, ValueError):
                print(f"Warning: Can't add final result {titlename}: {vary_x} into {tfile_path}!")
        else:
            combined_tdf = tdf

        combined_tdf.sort_values(by="Title", ascending=True, inplace=True)
        combined_tdf.to_csv(tfile_path, index=False)
        print(f"Data of final results of {titlename}: {vary_x} is saved into total results file!")

def load_final_csv_and_plot(fpath):
    """
    Load results with same variable type and plot
    """
    df=pd.read_csv(fpath)
    
    x=df["Varialbe Value"].tolist()

    subtitle=f"x-Axis: {x}"

    if isinstance(x[0],str):
        x=range(1,len(x)+1) 

    train_loss=df["Train Loss"].tolist()
    train_acc=df["Train Accuracy"].tolist()
    test_acc=df["Test Accuracy"].tolist()
    infer_acc=df["Infer Accuracy"].tolist()
    title=df["Title"].tolist()

    # print(x)
    # print(type(x))
    # print(train_loss)
    # print(type(train_loss))

    plot_acc(0,train_loss,train_acc,test_acc,infer_acc,f"{title[0]}_compare",xlist=x,sub_title=subtitle)

def load_sin_res(fpath):
    """
    Load one single training's results.
    """
    df=pd.read_csv(fpath)

    train_loss=df["Train Loss"].tolist()
    train_acc=df["Train Accuracy"].tolist()
    test_acc=df["Test Accuracy"].tolist()
    infer_acc=df["Infer Accuracy"].tolist()

    return [train_loss,train_acc,test_acc,infer_acc]

import os
import re

import os
import re

def get_next_demo_index(directory,match_word):
    """
    Find the next available index for files named 'demo*.csv' in the specified directory.

    Args:
        directory (str): The path to the directory to check.

    Returns:
        int: The next available index for a new file.
    """
    # List all files in the directory
    files = os.listdir(directory)
    
    demo_pattern = re.compile(match_word+r"(\d+)\.csv")
    indices = []
    
    for file in files:
        match = demo_pattern.match(file)
        if match:
            indices.append(int(match.group(1)))  # Extract the numeric index and convert to integer

    # If no matching files are found, start with index 1
    if not indices:
        return 1
    
    # Return the next index after the current maximum
    return max(indices) + 1


