import torch
from torch.utils.data import DataLoader, Dataset
import os
from utils import save_model, load_model, set_seed, make_checkpoint
import torch.nn as nn
import torch.optim as optim
import math
from torch.backends import cudnn

class InstaHide():
    """
    Wrapper class for training and testing with InstaHide instance encoding method (https://arxiv.org/abs/2010.02772)
    """

    def __init__(
            self,
            k:int=4,
            device:str="cpu",
            c:float=0.65,
            num_pred:int=10, 
            num_workers:int=8
            ) -> None:
        """
        Initialize parameters for all InstaHide methods.

        Arguments:
            - k: number of data samples to mix
            - device: device where to perform all computation
            - c: upper limit of the lambda values
            - num_pred: number of predictions to average during inference
            - num_workers: number of worker threads
        """
        set_seed()
        self.k = k
        self.device = device

        # Enforce valid values for c
        if c < 1/self.k:
            self.c = 1/self.k
        elif c > 1:
            self.c = 1
        else:
            self.c = c
        
        avg = 0
        std = self.c/3 # so that 99.7% of samples stay beetween plus/minus c
        self.normal = torch.distributions.normal.Normal(torch.Tensor([avg]), torch.Tensor([std]))

        self.num_pred = num_pred
        self.num_workers = num_workers

    def _get_loss(self, pred:torch.Tensor, target:torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.sum(- target * nn.functional.log_softmax(pred, dim=-1), 1))
    
    def _random_sign_flip(self, x:torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        with torch.device(self.device):
            mask = (2 * torch.randint(0, 2, (n, 1, h, w))) - 1

            return x*mask

    def _get_lambdas(self, n:int) -> torch.Tensor:
        with torch.device(self.device):
            lambdas = torch.abs(self.normal.sample((n, self.k))).squeeze(-1).to(self.device) # Sample from Normal distribution
            lambdas /= torch.sum(lambdas, dim=1).view(n, 1) # Normalize so that they sum to 1
            
            # Enforce the upper limit c
            invalid_idxs = torch.argwhere(lambdas > self.c)
            for i in invalid_idxs:
                l = torch.abs(self.normal.sample((self.k, ))).squeeze(-1)
                l /= torch.sum(l)
                while torch.any(l > self.c):
                    l = torch.abs(self.normal.sample((self.k, ))).squeeze(-1)
                    l /= torch.sum(l)
                lambdas[i[0], :] = l

            assert torch.all(lambdas <= self.c)
            assert torch.all(torch.isclose(torch.sum(lambdas, dim=1), torch.ones(1)))

            return lambdas
        

    def encode(
                self,
                private_data: torch.Tensor,                
                num_classes:int = None,
                private_labels: torch.Tensor = None,
                public_data: torch.Tensor=None
            ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode the private data and labels with InstaHide.
        
        Arguments:
            - private_data: the private data samples
            - num_classes: number of classes in the private dataset
            - private_labels: the private labels
            - public_data: a tensor of public images, must have the same size as private_data
            
        Return the encoded private data and the one-hot encoded private labels as a tuple.
        """

        with torch.no_grad() and torch.device(self.device):
            # Check validity of private_labels 
            if private_labels is not None:
                if num_classes is None:
                    Exception(f"Error: missing number of classes.")
                if num_classes < torch.max(private_labels):
                    Exception(f"Error: wrong nuFalsember of classes.")
                private_labels = private_labels.to(self.device)

            private_data=private_data.to(self.device)
            n, c, h, w = private_data.size()

            # Check public data has the right size
            if public_data is not None:
                if public_data.size() != private_data.size():
                    Exception(f"Error: public_data must be a tensor of size {private_data.size()}")
                else:
                    public_data = public_data.to(self.device)

            # Get lambdas
            if self.k > 1:
                lambdas = self._get_lambdas(n)
            else:
                lambdas = torch.ones(n, 1).to(self.device)

            x = torch.clone(private_data)*(lambdas[:, 0].reshape(n, 1, 1, 1)) # Broadcasting
            
            if private_labels is not None: # One-hot encode labels
                one_hot_labels = torch.nn.functional.one_hot(private_labels.long(), num_classes).type(torch.FloatTensor).to(self.device)
                assert one_hot_labels.size() == (n, num_classes)
                y = one_hot_labels*(lambdas[:, 0].reshape(n, 1)) # Broadcasting
            else:
                y = None

            # Mixup
            for i in range(1, self.k):
                idxs = torch.randperm(n) # Generate random permutation of indexes

                if i >= self.k//2 and public_data is not None:
                    x += public_data[idxs]*(lambdas[:, i].reshape(n, 1, 1, 1))
                else:
                    x += private_data[idxs]*(lambdas[:, i].reshape(n, 1, 1, 1))
                    if private_labels is not None:
                        y += one_hot_labels[idxs]*(lambdas[:, i].reshape(n, 1))
            

            # Random sign-flip
            x = self._random_sign_flip(x)

            return x, y


    def training(
        self,
        net: torch.nn.Module,
        training_set:Dataset,
        validation_set:Dataset,
        num_classes:int,
        num_epochs: int=100,
        batch_size:int = 128,
        lr: float=0.1,
        momentum: float=0.9,
        weight_decay: float=1e-4,
        path:str=os.getcwd(),
        public_dataset:Dataset=None,
        val_freq:int=15,
        resume_file:str=None
    ) -> float:
        
        """
        Train a model and save the best performing.

        Arguments:
            - net: network architecture to be trained
            - training_set: the training set
            - validation_set: the validation set
            - num_classes: the number of classes
            - num_epochs: number of epochs 
            - batch_size: the batch size
            - lr: learning rate
            - momentum: momentum
            - weight_decay: weight decay
            - path: path where to save all the files
            - public_dataset: public dataset to encode with Cross InstaHide
            - val_freq: after this many epochs a validation step will be performed
            - resume_file: path to a checkpoint file to be loaded with the function utils.load_model
    
        Return best model's validation accuracy.
        """

        # Make dataloaders
        tr_dataloader = DataLoader(training_set, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)    
        num_batches = math.floor(len(training_set)/batch_size)
        if public_dataset is not None:
            public_dataloader = DataLoader(public_dataset, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True)

        # Resume from checkpoint file if present
        if resume_file is not None and os.path.isfile(resume_file):
            data = load_model(resume_file)
            epoch = data["epoch"]
            lr = 0.1*lr + 0.9*0.5*lr*(1+math.cos(math.pi*epoch/num_epochs)) # Follow cosine annealing method
            weights = data["weights"]
            net.load_state_dict(weights)
            cur_epoch = epoch
            print(f'Resuming from checkpoint at epoch: {epoch}')
        else:
            cur_epoch = 0
            # Delete previous stats file
            if os.path.isfile(path + "/stats.csv"):
                os.remove(path + "/stats.csv")

        # Choose parameters to optimize
        parameters_to_optimize = net.parameters()

        # Define optimizer
        optimizer = optim.SGD(
            parameters_to_optimize, lr=lr, momentum=momentum, weight_decay=weight_decay
        )

        # Define scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer=optimizer, T_max=num_epochs-cur_epoch, eta_min=0.1*lr
        )
        
        # Send to device
        net = net.to(self.device)
        # Optimize
        cudnn.benchmark 

        # Train
        max_accuracy = 0
        try:
            for epoch in range(cur_epoch, num_epochs):
                print(
                    "Starting epoch {}/{}, LR = {}".format(
                        epoch + 1, num_epochs, scheduler.get_lr()
                    )
                )
                sum_losses = torch.zeros(1).to(self.device)

                # Iterate over the training dataset in batches
                for images, labels in tr_dataloader:
                    # Bring data over the device of choice
                    images = images.to(self.device)
                    labels = labels.to(self.device)

                    net.train()  # Sets module in training mode

                    optimizer.zero_grad()  # Zero-ing the gradients

                    # Encode with InstaHide
                    if public_dataset is not None:
                        public_data = next(iter(public_dataloader))[0].to(self.device)
                    else:
                        public_data = None

                    enc_data, target = self.encode(images, num_classes, labels, public_data)

                    # Forward pass to the network
                    pred = net(enc_data)

                    # Compute loss based on output and ground truth
                    loss = self._get_loss(pred, target)
                    sum_losses += loss

                    # Compute gradients for each layer and update weights
                    loss.backward()  # backward pass: computes gradients
                    optimizer.step()  # update weights based on accumulated gradients

                # Step the scheduler
                scheduler.step()

                # Compute and log the average loss over all batches
                avg_loss = sum_losses.item() / num_batches
                print(f"\tAverage loss = {avg_loss}")

                # Checkpoint
                make_checkpoint(net, path=path + "/checkpoint.pth", epoch=epoch+1, accuracy=None, lr=scheduler.get_last_lr()[-1])

                if ((epoch % val_freq) == 0 or num_epochs - epoch <= 10):
                    # Compute validation accuracy
                    acc = self.inference(net, validation_set, num_classes, batch_size, encoding_data=training_set)
                    print(f"\tValidation accuracy = {acc}")
                    
                    # Save the best model
                    if acc > max_accuracy:
                        save_model(net, path + "/best_model.pth", epoch, acc, scheduler.get_last_lr()[-1])
                        max_accuracy = acc

                    # Record stats
                    with open(path + "/stats.csv", "a") as f:
                        if epoch == 0:
                            f.write("epoch,avg_loss,accuracy\n")
                        f.write(f"{epoch},{avg_loss},{acc}\n")
        except KeyboardInterrupt:
            print(f"Early stopping at epoch {epoch+1}")
        
        return max_accuracy


    def inference(
            self,
            net: nn.Module,
            test_set:Dataset,
            num_classes:int,
            batch_size:int=128,
            model_path: str=None,
            encoding_data:Dataset=None
            ) -> float:
        """
        Test a model.

        Arguments:
            - net: network to be tested
            - test_set: the test set
            - num_classes: number of classes in the test set
            - batch_size: the batch size
            - model_path: path of a saved model to be loaded for inference
            - encoding_data: if present, this dataset is used for encoding the test set

        Return the accuracy on the test set
        """
        # Load model if available
        if model_path is not None:
            data = load_model(model_path)
            net.load_state_dict(data["weights"])

        net = net.to(self.device)
        net.train(False)  # Set Network to evaluation mode
        
        # Make dataloaders
        test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=self.num_workers, drop_last=True)
        n = len(test_set)
        if encoding_data is not None: 
            mix_dataloader = DataLoader(encoding_data, batch_size=batch_size, shuffle=True, num_workers=self.num_workers, drop_last=True,) #generator=torch.Generator(device=self.device))
        
        with torch.no_grad():
            
            running_corrects = 0
            for images, labels in test_dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                if encoding_data is not None:
                    mixing_data = next(iter(mix_dataloader))[0].to(self.device)
                    
                    # Make multiple predictions
                    output = torch.zeros(batch_size, num_classes).to(self.device)
                    for _ in range(self.num_pred):
                        # Mix-up each test sample with k-1 images from the training set
                        if self.k > 1:
                            lambdas = self._get_lambdas(batch_size)
                        else:
                            lambdas = torch.ones(batch_size, 1).to(self.device)

                        enc_images = images*(lambdas[:, 0].reshape(batch_size, 1, 1, 1))
                        for k in range(1, self.k):
                            idxs = torch.randperm(batch_size)
                            enc_images += mixing_data[idxs]*(lambdas[:, k].reshape(batch_size, 1, 1, 1))

                        # Random sign-flip
                        enc_images = self._random_sign_flip(enc_images)

                        # Do a forward pass and sum logits
                        output += net(enc_images)
                else:
                    output = net(images)

                # Take the max of logits as prediction
                pred = torch.argmax(output, dim=1)

                # Update Corrects
                running_corrects += torch.sum(pred == labels).item()

            # Calculate Accuracy
            accuracy = running_corrects / n

        # Save model with test accuracy if path available
        if model_path is not None:
            save_model(net, model_path, None, accuracy, None)

        net.train(True)

        return accuracy
    
# if __name__ == "__main__":
    
#     import time
#     import torchvision.transforms.functional as tf
#     import PIL

#     ih = InstaHide(k=4)
#     n = 128

#     x = torch.randn((1, 3, 32, 32))
#     enc_x = ih._random_sign_flip(x)


#     im = tf.to_pil_image(x.squeeze(0))
#     enc_im = tf.to_pil_image(enc_x.squeeze(0))
#     im.show()
#     enc_im.show()



#     # print(torch.sum(lambdas > 0).item()/(4*n))
#     # print(torch.sum(lambdas < 0).item()/(4*n))
    
#     pass