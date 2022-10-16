from __future__ import print_function, division
from atexit import register

from builtins import range
from builtins import object
from operator import mod
import os
import pickle as pickle

import wandb
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
from src.dataset import SimpleDataset

from .registry import optimier,Activation

class Solver(object):
    """
    A Solver encapsulates all the logic necessary for training models. The 
    Solver performs stochastic gradient descent using different update rules.
    The solver accepts both training and validataion data and labels so it can
    periodically check loss on both training and validation data to watch out 
    for overfitting.
    To train a model, We need to first construct a Solver instance, passing the
    model, dataset, and various options (learning rate, batch size, etc) to the
    constructor. Then call the train() method to run the optimization procedure 
    and train the model.
    After the train() method returns, model.params will contain the parameters
    that performed best on the validation set over the course of training.
    In addition, the instance variable solver.loss_history will contain a list
    of all losses encountered during training and the instance variables
    solver.train_acc_history and solver.val_acc_history will be lists of the
    accuracies of the model on the training and validation set at each epoch.

    Inputs:
      - X: Array giving a minibatch of input data of shape (N, 1)
      - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
        label for X[i].

      Returns:
      If y is None, run a test-time forward pass and return:
      - pred y.
      If y is not None, run a training time forward and backward pass and
      return a tuple of:
      - loss: Scalar giving the loss.
    """

    def __init__(self,model,data,logger,**kwargs) -> None:
        """"
        Construct a new Solver instance.
        - model : A Neural Network model inherit from torch.nn.Module
        - data  : A dictionary of training and validation data

        Optional arguments:
        - optimer : A string giving an update rule
        - lr : learning rate
        - batch_size : Size of minibatches used to compute loss and gradient
          during training
        - num_epochs: The number of epochs to run for during training
        - print_every: Integer; training losses will be printed every
          print_every iterations
        - num_train_samples: Number of training samples used to check training
          loss; default is 500; set to None to use entire training set.
        - num_val_samples: Number of validation samples to use to check val
          loss; default is None, which uses the entire validation set.
        - checkpoint_name: If not None, then save model checkpoints here every
          epoch 
        """
        self.model = model
        self.train_X = data["train_X"]
        self.train_y = data["train_y"]
        self.eval_X  = data["train_X"]
        self.eval_y  = data["train_y"]

        self.lr = kwargs.get("learning_rate",1e-5)
        self.bs = kwargs.get("batch_size",64)
        self.print_every = kwargs.get("print_every",20)
        self.num_train_samples = kwargs.get("num_train_samples",1000)
        self.num_val_samples = kwargs.get("num_val_samples",None)
        self.epochs = kwargs.get("epochs",100)
        self.logger = logger
        self.wandb = kwargs.get("wandb",True)

        optimizer = kwargs.get("optimer","adam")

        self.best_model = model

        self.loss = nn.MSELoss()
        self.optimizer = optimier[optimizer](self.model.parameters(),lr = self.lr)
        
        self._reset()


    def _reset(self):
        self.train_loss_his = []
        self.eval_loss_his = []
        self.best_val_acc = 0.0

    def _step(self,batch_X,batch_y):
        """
        Make a single gradient update. This is called by train() and should not
        be called manually.
        """
        # Compute loss 
        pred_y = self.model(batch_X)
        loss = self.loss(pred_y,batch_y)
        self.train_loss_his.append(loss)

        return loss

    def check_accuracy(self,X,y,num_samples=None,batch_size = 100):
        """
        Check accuracy of the model on the provided data.

        Inputs:
        - X: Array of data, of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,)
        - num_samples: If not None, subsample the data and only test the model
          on num_samples datapoints.
        - batch_size: Split X and y into batches of this size to avoid using
          too much memory.

        Returns:
        - acc: Scalar giving the fraction of instances that were correctly
          classified by the model.
        """
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
          mask = np.random.choice(N,num_samples)
          X = X[mask]
          y = y[mask]
          N = num_samples
        
        iterations = N // batch_size
        # if N % batch_size != 0 :
        #   iterations += 1
        
        y_preds = []
        for iter in range(iterations):
          start = iter * batch_size
          end = (iter + 1) * batch_size
          y_preds.append(self.model(X[start:end]))

        y_preds = np.hstack(y_preds)
        acc = np.mean((y_preds - y.numpy())**2)
        
        return acc

    def train(self):
        """
        Run optimization to train the model.
        """
        dataset = SimpleDataset(self.train_X,self.train_y)

        train_dataloader = DataLoader(dataset,batch_size=self.bs,shuffle=True)
        for iter in range(self.epochs):
          t = 0
          for batch_X,batch_y in train_dataloader:
            loss = self._step(batch_X,batch_y)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Maybe print training loss
            if t % self.print_every == 0:
              print(
                    "(Iteration %d / %d) loss: %f"
                    % (t + 1, self.epochs, self.train_loss_his[-1])
                    )
            t += 1
          # Check train and val accuracy on the first iteration, the last
          # iteration, and at the end of each epoch.
          with torch.no_grad():
            train_loss = self.check_accuracy(self.train_X,
                  self.train_y,num_samples = self.num_train_samples
            )
            eval_loss = self.check_accuracy(self.eval_X,
                  self.eval_y,num_samples = self.num_val_samples
            )
          self.train_loss_his.append(train_loss)
          self.eval_loss_his.append(eval_loss)

          print(f"(Iteration {iter} % {self.epochs}) train loss: {train_loss}, eval loss: {eval_loss}")

          if eval_loss < self.best_val_acc:
            for name,params in self.model.state_dict().items():
                self.best_model.state_dict()[name].copy_(params)
          
          self.logger.add_scalar("../result/" + "train_loss",train_loss,iter)
          self.logger.add_scalar("../result/" + "train_loss",eval_loss,iter)
          
          if self.wandb:
            wandb.log(train_loss,iter)
            wandb.log(eval_loss,iter)


