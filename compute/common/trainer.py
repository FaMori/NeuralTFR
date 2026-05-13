import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR, LambdaLR
from tqdm.auto import tqdm

from common.dataset import SlidingWindowDataset
from common.losses import MQLoss
from models.ENC_DEC_GRU import ENC_DEC_GRU

class Trainer():
    """
    Trainer class 
        Train a torch model
        Args:
            model(nn.Module): torch model to train
            loss_function(MQLoss|RMSELoss): loss function
                MQLoss: multi quantile loss function
            optimizer_config(dict): dictionary with optimizer configuration
                lr(float): learning rate
                type_schelduler(str): learning rate scheduler type
                    exponential: exponential decay
                        gamma(float): decay rate
                    linear: linear decay
                        end_factor(float): ending factor            
                weight_decay(float): weight decay regularization
            log(bool): log training process
            pbar(tqdm): tqdm progress bar instance
    """
    def __init__(self,
                 model:ENC_DEC_GRU,
                 loss_function:MQLoss,
                 optimizer_config:dict,
                 log:bool=True,
                 pbar:tqdm=None
                 ) -> None:
        
        self.model = model
        self.loss_function = loss_function
        
        self.optimizer_config = optimizer_config

        self.train_losses, self.valid_losses = {}, {}
        self.last_epoch, self.last_step = 0, 0
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=optimizer_config['lr'], 
                                          weight_decay=optimizer_config['weight_decay'])
        
        self.scheduler = None
        self.log = log
        self.pbar = pbar

    def __call__(self,
                 train_dataset:SlidingWindowDataset,
                 valid_dataset:SlidingWindowDataset,
                 batch_size:int,
                 epochs:int,
                 max_steps:int,
                 valid_epochs:int,
                 stop_patience:int
                ) -> nn.Module:
        """
        Call method
            Compute the loss and optimize the model parameters
            Args:
                train_dataset(SlidingWindowDataset): training dataset
                valid_dataset(SlidingWindowDataset): validation dataset
                batch_size(int): batch size
                epochs(int): number of epochs
                max_steps(int): maximum number of training steps
                valid_epochs(int): number of epochs to validate
                stop_patience(int): early stopping patience
            Returns:
                model(nn.Module): trained model
        """
        early_stopping = EarlyStopping(patience=stop_patience) if stop_patience > 0 else None

        self._set_scheduler(self.optimizer_config)

        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True) 
        
        step = 0
        max_steps_reached = False
        for epoch in range(epochs):

            self.model.train()

            valid_epoch = valid_epochs > 0 and epoch % valid_epochs == 0

            if hasattr(self.model, 'tf_config'):
                self.model.tf_rate = self.model._compute_tf_rate(epoch=epoch)

            epoch_loss = 0.0
            for model_input, _ in train_dataloader:

                if step >= max_steps and max_steps > 0:
                    if self.log and self.pbar:
                        self.pbar.set_description(f"Model (Max Steps)")
                    max_steps_reached = True
                    break
                
                step += 1     
                loss = self._train_step(model_input)
                epoch_loss += loss.item()

            if max_steps_reached: 
                break
            
            self.last_epoch, self.last_step = epoch, step
            avg_loss = epoch_loss / len(train_dataloader)
            self.train_losses[epoch] = avg_loss

            if self.scheduler:
                self.scheduler.step()
            
            postfix_metrics = {
                'Train Loss': f"{avg_loss:.4f}",
                'Learning Rate': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            }

            if valid_epoch:
                val_loss = self._eval_step(valid_dataset.model_inputs)
                self.valid_losses[epoch] = val_loss
                postfix_metrics['Valid Loss'] = f"{val_loss:.4f}"

                if early_stopping:

                    early_stopping(val_loss)

                    if early_stopping.early_stop:
                        if self.log and self.pbar:
                            self.pbar.set_description(f"Model (Early Stop)")
                        break
            
            if self.log and self.pbar:
                self.pbar.set_postfix(postfix_metrics)
                self.pbar.update(1)

        if self.log and self.pbar:
            self.pbar.close()

        return self.model

    def _train_step(self,
                    model_input:dict[str, torch.Tensor],
                    max_grad_norm:float=5.0
                    ) -> torch.Tensor:       
        """
        Train step
            Compute the loss and optimize the model parameters
            Args:
                model_input(dict): dictionary with model inputs
                max_grad_norm(float): maximum gradient norm for clipping
            Returns:
                loss(torch.Tensor): computed loss
        """     
            
        self.optimizer.zero_grad()

        y = model_input['y']
        y_hat = self.model(model_input)
        loss = self.loss_function(y, y_hat) 

        loss.backward()

        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=max_grad_norm)
        self.optimizer.step()

        return loss
    
    def _eval_step(self,
                   model_input:dict[str, torch.Tensor]
                  ) -> tuple[torch.Tensor, np.float64, float]:      
        """
        Evaluation step 
            Compute the loss and evaluation metric
            Args:
                model_input(dict): dictionary with model inputs
            Returns:
                loss(np.float64): computed loss value
        """
        
        self.model.eval()     
        with torch.no_grad():

            y = model_input['y']
            if hasattr(self.model, 'tf_config'):
                self.model.tf_rate = 0.0
            y_hat = self.model(model_input)                 
            loss = self.loss_function(y,y_hat).item()
                        
        return loss
            
    def _set_scheduler(self,
                       scheduler_config:dict
                       ) -> None:

        scheduler_type = scheduler_config.get('type_scheduler', 'Step').lower()

        if scheduler_type == 'linear':
            max_epochs = self.optimizer_config.get('max_epochs', 100)
            end_factor = scheduler_config.get('end_factor', 0.1)
            end_factor = max(1e-6, min(1.0, end_factor))
            self.scheduler = LambdaLR(self.optimizer, lr_lambda=lambda epoch: 1.0 - epoch / (max_epochs - 1) * (1.0 - end_factor))

        elif scheduler_type == 'exponential':
            gamma = scheduler_config.get('gamma', 0.9)
            self.scheduler = ExponentialLR(self.optimizer, gamma=gamma)

class EarlyStopping():
    """
    Early stopping class
        Stop training if the validation loss does not improve for a given number of epochs
        Args:
            patience(int): number of epochs to wait before stopping
            min_delta(float): minimum change in the monitored quantity to qualify as an improvement
    """

    def __init__(self, patience=2, min_delta=1e-3):
 
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True