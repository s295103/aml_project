"""This file contains useful functions"""
import warnings
warnings.simplefilter("ignore", UserWarning)
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import Subset, Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100
import os
import torch.optim as optim
from torch.backends import cudnn
import numpy as np


def save_model(model: nn.Module, path: str, epoch: int = None, accuracy: float = None, lr: float = None):
    """
    Saves the state of a model as a dictionary.

    Arguments:
        - model: the model whose state needs to be saved
        - path: path in which to save the state 
        - epoch: epoch number, useful for checkpointing
        - accuracy: accuracy of current model
        - lr: learning rate, useful for checkpointing
    """
    state = {'weights': model.state_dict(),
             'accuracy': accuracy,
             'epoch': epoch,
             'lr': lr,
             }
    torch.save(state, path)


def load_model(path: str) -> dict:
    """
    Load a model's state; it's the counterpart of the function "save_model".
    
    Arguments
        - path: path of the state

    Return a dictionary with the following keys:
        - weights: model's parameters, to be loaded with the function "nn.Module.load_state_dict"
        - accuracy: model's accuracy
        - epoch: epoch number, useful for checkpointing
        - lr: learning rate, useful for checkpointing
    """
    if torch.cuda.is_available():
        map_location = torch.device("cuda")
    else:
        map_location = torch.device("cpu")
    return torch.load(path, map_location)


def model_size(net: nn.Module) -> int:
    """
    Return a model's number of parameters.
    """
    tot_size = 0
    for param in net.parameters():
        tot_size += param.size()[0]
    return tot_size

def _make_checkpoint(net:nn.Module, path:str, epoch:int=None, accuracy:float=None, lr:float=None, verbose=False) -> None:
    save_model(net, path, epoch, accuracy, lr)
    if verbose:
        print(f"Checkpoint at epoch {epoch} saved at path {path} ")


def _get_validation_set(training_data: Dataset, val_fraction: float) -> tuple[Subset, Subset]:
    """
    Extract a validation set from the training data.
    
    Arguments
        - training_data: the training data
        - val_fraction: fraction of training_data to use as validation set

    Return a tuple of Subset containing the (remaining) training set and the validation set
    """

    # Get training dataset length
    tr_data_len = len(training_data)

    # Shuffle indexes
    shuffled_indexes = torch.randperm(tr_data_len)

    # Partition indexes
    train_indexes = shuffled_indexes[0: int(tr_data_len * (1 - val_fraction))]
    val_indexes = shuffled_indexes[int(
        tr_data_len * (1 - val_fraction)): tr_data_len]

    tr_set = Subset(training_data, train_indexes)
    val_set = Subset(training_data, val_indexes)

    return tr_set, val_set


def cifar_processing(
        cifar100: bool = False,
        val_ratio: float = 0,
        root: str = os.getcwd()
        ) -> tuple[Dataset, Dataset, Dataset]:
    """
    Load and preprocess the CIFAR dataset.
    
    Arguments
        - cifar100: if True, load CIFAR100, otherwise load CIFAR10
        - val_ratio: fraction of the training set to use as validation set, if 0 does not make a validation set
        - root: path where to download the dataset

    Return a tuple containing the training set, the validation set and the test set, as Dataset objects.
    """

    # Define transforms
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(0.5),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    test_transforms = train_transforms

    # Choose the right CIFAR
    if cifar100:
        data = CIFAR100
    else:
        data = CIFAR10

    # Load dataset
    train_dataset = data(root, True, train_transforms, download=True)
    test_dataset = data(root, False, test_transforms, download=True)

    # Get validation set
    if val_ratio > 0:
        train_dataset, val_dataset = _get_validation_set(train_dataset, val_ratio)
    else:
        val_dataset = None

    return train_dataset, val_dataset, test_dataset

def training(
    name:str,
    net: torch.nn.Module,
    training_set: Dataset,
    test_set: Dataset,
    num_epochs: int = 100,
    batch_size:int=128,
    lr: float=0.1,
    momentum: float=0.9,
    weight_decay: float=1e-4,
    path:str=os.getcwd(),
    device:str="cpu",
    num_workers:int = 4,
    test_freq:int=15,
    resume_file:str=None
) -> float:
    
    """
    Train a model and save the best performing.

    Arguments:
        - name: name of the model, all files will be prepended with this
        - net: network architecture to be trained
        - training_set: training set
        - test_set: test set
        - num_epochs: number of epochs to train the net, if <= 0 it will train forever
        - batch_size: batch size
        - lr: learning rate
        - momentum: momentum
        - weight_decay: weight decay
        - path: path of the folder where to save all the files
        - device: all operations on the model will be performed on this device
        - num_workers: number of worker threads
        - test_freq: inference on the test set will be performed every test_freq epochs
        - resume_file: path to a checkpoint file to be loaded with the function utils.load_model

    Return the maximum accuracy.
    """
    
    ############################
    # Note: epochs starts at 1 #
    ############################

    filename = f"{path}/{name}"


    if test_freq <= 0:
        raise Exception("Error: test frequency must be an integer greater than zero")
    if test_freq > num_epochs and num_epochs > 0:
        raise Exception(f"Error: test frequency must be lower than {num_epochs}")

    # Test for convergence every num_epochs
    if num_epochs <= 0:
        max_epochs = test_freq
    else:
        max_epochs = num_epochs

    # Make dataloader
    train_dataloader = DataLoader(
        training_set, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
        
    # Resume from checkpoint file if present
    if resume_file is not None and os.path.isfile(resume_file):
        data = load_model(resume_file)
        epoch = data["epoch"]
        if num_epochs <= 0:
            lr = data["lr"]
        else:
            lr = 0.1*lr + 0.9*0.5*lr*(1+np.cos(np.pi*epoch/num_epochs)) # Follow cosine annealing method
        weights = data["weights"]
        net.load_state_dict(weights)
        print(f'Resuming from checkpoint at epoch: {epoch}')
    else:
        epoch = 1
        # Delete previous stats file
        if os.path.isfile(f"{path}/{name}_stats.csv"):
            os.remove(f"{path}/{name}_stats.csv")

    # Define loss function
    criterion = nn.CrossEntropyLoss()

    # Choose parameters to optimize
    parameters_to_optimize = net.parameters()

    # Define optimizer
    optimizer = optim.SGD(
        parameters_to_optimize, lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    # Define scheduler
    if num_epochs <= 0:
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer=optimizer, T_0=50, T_mult=1, eta_min=0.1*lr
        )
    else:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=num_epochs-epoch, eta_min=0.1*lr
        )


    # Send to device
    net = net.to(device)
    # Optimize
    cudnn.benchmark 

    # Train
    
    max_accuracy = 0
    accuracy_per_epoch = []
    try:
        while not _termination(epoch, max_epochs):
            if num_epochs <= 0:
                print(f"Epoch {epoch}, LR = {scheduler.get_last_lr()[0]}")
            else:
                print(f"Epoch {epoch}/{max_epochs}, LR = {scheduler.get_last_lr()[0]}")

            avg_loss = step(net, train_dataloader, optimizer, criterion, scheduler, device)
            
            print(f"\tLoss = {avg_loss}")

            # Make new checkpoint and remove the previous one
            new_checkpoint = f"{filename}_checkpoint_{epoch}.pth"
            if epoch > 1:
                old_checkpoint = f"{filename}_checkpoint_{epoch-1}.pth"
                if os.path.isfile(old_checkpoint):
                    os.remove(old_checkpoint)
            _make_checkpoint(net, path=new_checkpoint, epoch=epoch, accuracy=None, lr=scheduler.get_last_lr()[-1])

            # Compute accuracy
            if epoch == 1 or (epoch % test_freq) == 0 or (num_epochs > 0 and num_epochs - epoch <= 10):
                acc = testing(net, test_set, batch_size, device, num_workers)
                print(f"\tAccuracy = {acc}")
                accuracy_per_epoch.append(acc)
                
                # Save the best model
                if acc > max_accuracy:
                    save_model(net, f"{filename}_best_model.pth", epoch, acc, scheduler.get_last_lr()[-1])
                    max_accuracy = acc

                # Record stats
                with open(f"{filename}_stats.csv", "a") as f:
                    if epoch == 1:
                        f.write("epoch,avg_loss,accuracy\n")
                    f.write(f"{epoch},{avg_loss},{acc}\n")
                
                if num_epochs <= 0:
                    max_epochs += test_freq         
            
            epoch += 1
    
    except KeyboardInterrupt:
        print(f"Training interrupted at epoch {epoch}")

    return max_accuracy


def _termination(epoch:int, max_epochs:int) -> bool:
    return epoch > max_epochs

def step(net:nn.Module, 
         training_data:DataLoader, 
         optimizer:optim.Optimizer, 
         criterion:nn.Module, 
         scheduler:optim.lr_scheduler.LRScheduler, 
         device:str="cuda" if torch.cuda.is_available() else "cpu", ):
    
    sum_losses = torch.zeros(1).to(device)
    
    num_batches = len(training_data)

    # Iterate over the training dataset in batches
    for images, labels in training_data:
        # Bring data over the device of choice
        images = images.to(device)
        labels = labels.to(device)

        net.train()  # Sets module in training mode

        optimizer.zero_grad()  # Zero-ing the gradients

        # Forward pass to the network
        outputs = net(images)

        # Compute loss based on output and ground truth
        loss = criterion(outputs, labels)
        sum_losses += loss

        # Compute gradients for each layer and update weights
        loss.backward()  # backward pass: computes gradients
        optimizer.step()  # update weights based on accumulated gradients

    # Step the scheduler
    scheduler.step()


    # Compute and log the average loss over all batches
    avg_loss = sum_losses.item() / num_batches

    return avg_loss

def testing(
        net: nn.Module, 
        test_set: Dataset, 
        batch_size:int=128, 
        device:str="cpu", 
        num_workers:int=4, 
        model_path: str=None):
    """
    Test a model.

    Arguments:
        - net: network to be tested
        - test_set: the test dataset
        - batch_size: batch size
        - device: all operations on the model will be performed on this device        
        - num_workers: number of worker threads
        - model_path: path of a saved model to be loaded for inference

    Return the accuracy on the test set
    """
    # Make dataloader
    test_dataloader = DataLoader(
        test_set, batch_size=batch_size, num_workers=num_workers,shuffle=False)
    n = len(test_set)

    # Load model if available
    if model_path is not None:
        data = load_model(model_path)
        net.load_state_dict(data["weights"])

    net = net.to(device)
    net.train(False)  # Set Network to evaluation mode

    running_corrects = 0
    for images, labels in test_dataloader:

        images = images.to(device)
        labels = labels.to(device)

        # Forward Pass
        outputs = net(images)

        # Get predictions
        _, preds = torch.max(outputs.data, 1)

        # Update Corrects
        running_corrects += torch.sum(preds == labels.data).data.item()

    # Calculate Accuracy
    accuracy = running_corrects / n

    # Save model with test accuracy if path available
    if model_path is not None:
        save_model(net, model_path, None, accuracy, None)

    net.train(True)

    return accuracy

def read_stats(file_path:str) -> tuple[list]:
    """
    Read stats.csv file created by the function training.
    Return a tuple of lists containing the epochs, the losses and the accuracies.
    """
    with open(file_path, "r") as f:
        f.readline()
        lines = f.readlines()
    values = [line.rstrip("\n").split(",") for line in lines ]
    stats_tuples = [(int(v[0]), float(v[1]), float(v[2])) for v in values ]
    epochs, loss, acc = [list(i) for i in zip(*stats_tuples)]
    return epochs, loss, acc

def set_seed(seed: int = 42) -> None:
    """
    Code from https://wandb.ai/sauravmaheshkar/RSNA-MICCAI/reports/How-to-Set-Random-Seeds-in-PyTorch-and-Tensorflow--VmlldzoxMDA2MDQy
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

