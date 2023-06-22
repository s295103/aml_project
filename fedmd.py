from architectures import *
import torch.optim as optim
import pickle
from torch.utils.data import Dataset, Subset, DataLoader
import torch.nn as nn
import torch
from utils import save_model, load_model
import os

class Client():
    def __init__(self,
                 name: str,
                 model: nn.Module,
                 private_data: Dataset = None,
                 ):
        self.name = name
        self.model = model
        self.private_data = private_data
        self.optimizer = None
        self.criterion = None
        self.scheduler = None
        self.scheduler_kwargs = None
        self.scheduler_obj = None
        self.device = None
        self.lr = None
        self.weigt_decay = None
        self.momentum = None
        self.batch_size = None
        self.path = None
        self.num_workers = None
        self.max_accuracy = None
        self.prediction = None

    def init_coop_training(
            self,
            optimizer: optim.Optimizer,
            criterion: nn.Module,
            scheduler: optim.lr_scheduler.LRScheduler = None,
            scheduler_kwargs:dict=None,
            device: str = "cuda" if torch.cuda.is_available() else "cpu",
            lr: float = 0.1,
            weight_decay: float = 1e-4,
            momentum: float = 0.9,
            batch_size: int = 128,
            path: str = os.getcwd(),
            num_workers: int = 8, 
            checkpoint_file_path:str=None) -> None:

        self.optimizer = optimizer
        self.criterion = criterion
        if scheduler_kwargs is not None and scheduler is not None:
            self.scheduler_kwargs = scheduler_kwargs
            self.scheduler_obj = scheduler
            self.scheduler = self.scheduler_obj(**self.scheduler_kwargs)
        self.device = device
        self.lr = lr
        self.weigt_decay = weight_decay
        self.momentum = momentum
        self.batch_size = batch_size
        self.path = path
        self.num_workers = num_workers

        # Resume from checkpoint
        if checkpoint_file_path is not None and \
            os.path.isfile(checkpoint_file_path):
            model_data = load_model(checkpoint_file_path)
            self.model.load_state_dict(model_data["weights"])
            self.last_round = model_data["epoch"]
        else:
            self.last_round = None

        self.max_accuracy = 0
        self.model.to(device) 

    
    def testing(self, test_set:Dataset) -> float:
        test_dl = DataLoader(test_set, self.batch_size, shuffle=False, num_workers=self.num_workers)
        self.model.train(False)
        with torch.no_grad():
            correct = 0
            for x, y in test_dl:
                x = x.to(self.device)
                y = y.to(self.device)
                pred = self.model(x)
                
                correct += torch.sum(pred.argmax(dim=1) == y)
            
            self.model.train(True)
            return correct / len(test_set)
                

    def checkpoint(self, round:int, accuracy:float, path:str, verbose:bool=False) -> None:
        lr = self.scheduler.get_last_lr() if self.scheduler is not None else self.lr
        
        if accuracy > self.max_accuracy:
            save_model(self.model, f"{path}/{self.name}_best_model.pth", round, accuracy, lr)
            self.max_accuracy = accuracy
            if verbose:
                print(f"{self.name} best model saved at path {path}")
        
        save_model(self.model, f"{path}/{self.name}_checkpoint_{round}.pth", round, accuracy, lr)
        if verbose:
            print(f"{self.name} checkpoint at round {round} saved at path {path}")
    
    def private_training(self, num_epochs:int, test_set:Dataset) -> tuple[float, float]:
        tr_dl = DataLoader(self.private_data, self.batch_size, True, num_workers=self.num_workers)
        print(f"{self.name} private training")
        try:
            for e in range(num_epochs):
                sum_loss = 0
                self.model.train(True)
                for x, y in tr_dl:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    self.optimizer.zero_grad()

                    y_pred = self.model(x)

                    loss = self.criterion(y_pred, y)
                    sum_loss += loss.item()

                    loss.backward()
                    self.optimizer.step()
                
                if self.scheduler is not None:
                    self.scheduler.step()
                avg_loss = sum_loss / len(tr_dl)
                print(f"Epoch {e}, avg_loss = {avg_loss}")
        except KeyboardInterrupt:
            print(f"Private training interrupted at epoch {e}")
        
        # Testing
        acc = self.testing(test_set)
        print(f"{self.name} accuracy = {100*acc:.1f} %")
        
        # Reset scheduler
        if self.scheduler is not None and \
            self.scheduler_obj is not None and \
                self.scheduler_kwargs is not None:
            self.scheduler = self.scheduler_obj(**self.scheduler_kwargs)

        return avg_loss, acc
    
    def predict_consensus(self, x:torch.Tensor) -> torch.Tensor:
        self.model.train(True)
        x = x.to(self.device)
        self.optimizer.zero_grad()
        self.prediction = self.model(x)
        return self.prediction

    def digest_consensus(self, target:torch.Tensor, coop_lr:float) -> None:
        if target.size() != self.prediction.size():
            raise Exception(f"Error: target size {target.size()} and prediction size {self.prediction.size()} must match")
        
        # Swap private and cooperative training learning rate 
        pr_lr = self.optimizer.param_groups[0]["lr"]
        self.optimizer.param_groups[0]["lr"] = coop_lr
        
        # Backward pass
        target = target.to(self.device)
        loss = self.criterion(self.prediction, target) 
        loss.backward()
        self.optimizer.step()
        
        # Reset last prediction and lr
        self.prediction = None
        self.optimizer.param_groups[0]["lr"] = pr_lr


class Server():
    def __init__(self, 
            clients: list, 
            train_set: Dataset, 
            test_set:Dataset, 
            num_classes_pr_data:int,
            lr:float,
            max_rounds: int = 100, 
            priv_train_epochs:int=25, 
            pub_train_epochs:int=5, 
            dataset_size: int = 10_000, 
            batch_size:int=128, 
            num_workers:int=8, 
            device:str="cuda" if torch.cuda.is_available() else "cpu",
            path:str=os.getcwd()
            ) -> None:
        self.clients = clients
        self.num_clients = len(clients)
        self.num_classes_pr_data = num_classes_pr_data
        self.max_rounds = max_rounds
        self.lr = lr
        self.test_set = test_set
        self.priv_train_epochs = priv_train_epochs
        self.pub_train_epochs = pub_train_epochs
        self.dataset_size = dataset_size
        self.train_set = train_set
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.device = device
        self.path = path
        self.stats = dict(
            loss=[[] for _ in range(self.num_clients)],
            acc=[[] for _ in range(self.num_clients)]
        )

    def coop_training(self) -> dict:
        # Check for resumed clients
        last_rounds = [client.last_round for client in self.clients.values()]
        if any([l is None for l in last_rounds]): # If any client has not resumed from checkpoint, starts rounds from 0
            min_round = 0
        else: # If all clients have resumed from checkpoint, choose the minimum last round number
            min_round = min(last_rounds)
                
        try:
            for r in range(min_round, self.max_rounds):
                print(f"Round {r}")
                for _ in range(self.pub_train_epochs):
                    self.coop_step()
                    self.ind_step(r)
        except KeyboardInterrupt:
            print(f"Training interrupted at round {r}")

        pickle.dump(self.stats, open(f"{self.path}/stats.p", "wb"))
        return self.stats

    def coop_step(self):
        sampled_train_set = sample_dataset(self.train_set, size=self.dataset_size)
        tr_dl = DataLoader(sampled_train_set, self.batch_size, shuffle=True, num_workers=self.num_workers)
        for x, _ in tr_dl:
            logits = torch.zeros(1).to(self.device).detach() # Communication to the server + aggregation
            for c in self.clients.values(): 
                logits = logits + c.predict_consensus(x).to(self.device)
            
            mean = logits / self.num_clients
            std = torch.sqrt(torch.sum(torch.Tensor([torch.square(mean-c.prediction) for c in self.clients.values()])))
            print(f"Logits std = {std}")
            for n, c in self.clients.items(): # Server distributes consensus to the clients
                c.digest_consensus(mean.detach(), self.lr)  # Client digest consesus
                for p in c.model.parameters():
                    if torch.any(torch.isnan(p)):
                        raise Exception(f"Error: client {n} has NaN parameters")

    def ind_step(self, round:int):
        for i, c in enumerate(self.clients.values()):
            # Train on private data for some epochs
            loss, acc = c.private_training(self.priv_train_epochs, self.test_set)

            # Record stats for each client
            self.stats["acc"][i].append(acc)
            self.stats["loss"][i].append(loss)

            # Checkpoint the training so far
            c.checkpoint(round, acc, self.path)


def partition_dataset(data: Dataset, num_partitions: int, alpha: float, batch_size: int = 128) -> tuple[list, list]:
    """
    For each class in the dataset, distribute the data points to each partition proportionally to a 
    fraction sampled from the Dirichlet distribution.
    The lower the alpha, the higher is the inequality of classes distributed among the partitions.

    Return a tuple of two elements:
        the first is the list of partitions containing the indexes of the data samples 
        the second is a dict with two keys:
            - "class_std" indexes a list of standard deviation values for each class
            - "class_comp" indexes a dict like {<partition_id>:<list of samples of each class in this partition>}
    """

    # Put a cap on alpha values to avois numerical problems
    if alpha < 1e-3:
        alpha = 1e-3
    elif alpha > 1e9:
        alpha = 1e9

    # Get targets and number of classes
    targets = torch.Tensor(data.targets)
    num_classes = len(data.classes)

    # Initialize stats
    stats = {}
    stats["class_std"] = []
    stats["class_comp"] = {p: [] for p in range(num_partitions)}

    # Initialize Dirichlet distribution with all the parameters set to alpha
    dir = torch.distributions.dirichlet.Dirichlet(
        torch.Tensor([alpha]).repeat(num_partitions))

    # Distribute the data samples among the partitions
    partitions = [[] for _ in range(num_partitions)]
    for t in range(num_classes):
        data_idxs = torch.where(targets == t)[0]
        num_samples = len(data_idxs)

        frac = dir.sample()*num_samples
        cum_frac = torch.cumsum(frac, dim=0).int()

        for i in range(num_partitions):
            if i == 0:
                idxs = data_idxs[0:cum_frac[i]].tolist()
            else:
                idxs = data_idxs[cum_frac[i-1]:cum_frac[i]].tolist()

            partitions[i] += idxs

            # Record the number of samples of this class for this partititon
            stats["class_comp"][i].append(len(idxs))

        # Record the std for this class
        stats["class_std"].append(torch.std(frac).item())

    return partitions, stats


def make_fc_layer(in_feats: int, out_feats: int, hidden_feats: list = None) -> nn.Module:
    if hidden_feats is None or len(hidden_feats) == 0:
        return nn.Sequential(
            nn.BatchNorm1d(in_feats),
            nn.Linear(in_feats, out_feats)
        )
    else:
        layers = []
        layers.append(nn.BatchNorm1d(in_feats))
        layers.append(nn.Linear(in_feats, hidden_feats[0]))
        for i in range(0, len(hidden_feats)-1):
            layers.append(nn.BatchNorm1d(hidden_feats[i]))
            layers.append(nn.Linear(hidden_feats[i], hidden_feats[i+1]))
        layers.append(nn.BatchNorm1d(hidden_feats[-1]))
        layers.append(nn.Linear(hidden_feats[-1], out_feats))
        return nn.Sequential(*layers)


def sample_dataset(data: Dataset, size: int) -> Dataset:
    if size > len(data):
        raise Exception(f"Error: maximum size is {len(data)}")
    else:
        idxs = torch.randint(len(data), (size, ))
        return Subset(data, idxs)
    

def create_clients(num_classes:int=10) -> dict:
    clients = {}

    clients_names = ["resnet20gn", "shufflenetbig", "shufflenetsmall","densenet20","densenet10"]
    clients_models = [
        ResNet(BasicBlock, [3, 3, 3], num_classes=num_classes, groups=2, norm_layer="gn"),
        ShuffleNetV2([4, 8, 4], [32, 64, 128, 256, 512], num_classes=num_classes, groups=2),
        ShuffleNetV2([2, 4, 2], [16, 32, 64, 128, 256], num_classes=num_classes, groups=2),
        DenseNet(12, 20, 1, 10, False, 2),
        DenseNet(12, 10, 1, 10, False, 2),
    ]

    for name, model in zip(clients_names, clients_models):
        clients[name] = Client(name, model)

    return clients


def load_clients(path:str) -> dict:
    return pickle.load(open(f"{path}/clients.p", "rb"))


def save_clients(clients:dict, path:str) -> None:
    pickle.dump(clients, open(f"{path}/clients.p", "wb"))
