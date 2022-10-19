import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import numpy as np
import argparse
import wandb
import pandas as pd
from tensorboardX import SummaryWriter
from src.Solver import Solver
from src.registry import optimier,Activation

from src.model import Model

parser = argparse.ArgumentParser("Train a simple Neural Network")
parser.add_argument("--save-path",type=str,nargs="?",default = "./result",
                    help = "Please enter the directory of saving results")
parser.add_argument("data_path",type=str,nargs="?",default = "./data",
                    help = "The path to read data")
parser.add_argument("--gpuid",type=int,nargs="?",default = 0,help="Use cuda or not")
parser.add_argument("--epochs",type=int,nargs="?",default = 100,help="train epochs")
parser.add_argument("--alias",type=str,nargs="?",default=0,
                    help = "Please enter the alias")
parser.add_argument("--wandb",action="store_false")
parser.add_argument("--batch-size","-bs",type=int,nargs="?",default = 64,
                    help = "train batch size")
parser.add_argument("--learning-rate","-lr",nargs="?",default=1e-3,
                    help = "learning rate")
parser.add_argument("--optimer",type=str,nargs="?",default="adam",
                    help = "optimier to train")
parser.add_argument("--hidden-dim",type=int,nargs="?",default = 16,
                    help = "width size of the NN")
parser.add_argument("--n-layers",type=int,nargs="?",default=3,
                    help = "depth size of the NN")
parser.add_argument("--activation",type = str,nargs="?",default = "Relu",
                    help = "nonlinear activation function")


argv = parser.parse_args(args=[])

if torch.cuda.is_available():
    torch.cuda.set_device(argv.gpuid)
device = "cuda" if torch.cuda.is_available() else "cpu"

log_name = "-".join([argv.optimer,f"{argv.batch_size}",f"{argv.learning_rate}",argv.activation,
                f"{argv.n_layers}",f"{argv.hidden_dim}",f"{argv.alias}"])
if argv.wandb:
    wandb.init(
        project = 'deeplearning',
        name=log_name,
        group=log_name.split('_')[0],
        save_code=True
    )
    wandb.config.update(argv)
    wandb.run.log_code('.')

if argv.save_path[-1] == "/":
    save_path = argv.save_path
else:
    save_path = argv.save_path + "/"

if argv.data_path[-1] == "/":
    data_path = argv.data_path
else:
    data_path = argv.data_path + "/"

if "model_save" not in os.listdir(save_path):
    os.mkdir(save_path + "model_save")
if "tensorboard" not in os.listdir(save_path):
    os.mkdir(save_path + "tensorboard")
if log_name not in os.listdir(save_path + "model_save"):
    os.mkdir(save_path + "model_save/" + log_name)
if log_name not in os.listdir(save_path + "tensorboard/"):
    os.mkdir(save_path + "tensorboard/" + log_name)
else:
    path = save_path + "tensorboard/" + log_name
    for f in os.listdir(path):
        file_path = os.path.join(path, f)
        if os.path.isfile(file_path):
            os.remove(file_path)


logger = SummaryWriter(save_path + "tensorboard/" + log_name)

# optim = optimier[argv.optimer]

activation = Activation[argv.activation]

if "data.csv" not in os.listdir(data_path):
    raise RuntimeError("Please make sure have generated data")

data = pd.read_csv(data_path + "data.csv")

train_X = torch.tensor(data.loc[:7999,"x"].values).unsqueeze(1)
train_y = torch.tensor(data.loc[:7999,"y"].values).unsqueeze(1)

eval_X = torch.tensor(data.loc[8000:9499,"x"].values).unsqueeze(1)
eval_y = torch.tensor(data.loc[8000:9499,"y"].values).unsqueeze(1)

model = Model(argv,activation).double()

data = dict(train_X = train_X,train_y = train_y,
            eval_X  = eval_X, eval_y  = eval_y )

solver = Solver(model,data,logger,**vars(argv))

solver.train()

torch.save({"model_state_dict":solver.best_model.state_dict()},
            save_path + "model_save/" + log_name + "/model.pt")
print(True)




