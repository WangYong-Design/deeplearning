import os
import math
from re import I
import pandas as pd
import numpy as np

num_data  = 10000
num_eval  = 1000
num_test  = 100


save_path = "./data"

cols_idx = ["x","y"]

data_arr = np.zeros((num_data,2))

idx = 0
for x in np.random.uniform(0,4*math.pi,size = num_data):
    y = math.sin(x) + math.exp(-x)
    data_arr[idx][0] = x
    data_arr[idx][1] = y
    idx += 1

# idx = 0
# train_arr = np.zeros((num_train,2))
# for x in np.linspace(0,4*math.pi,num_train):
#     y = math.sin(x) + math.exp(-x)
#     train_arr[idx][0] = x
#     train_arr[idx][1] = y
#     idx += 1

# idx = 0
# eval_arr = np.zeros((num_eval,2))
# for x in np.linspace(0,4*math.pi,num_eval):
#     y = math.sin(x) + math.exp(-x)
#     eval_arr[idx][0] = x
#     eval_arr[idx][1] = y
#     idx += 1

# idx = 0
# test_arr = np.zeros((num_test,2))
# for x in np.linspace(0,4*math.pi,num_test):
#     y = math.sin(x) + math.exp(-x)
#     test_arr[idx][0] = x
#     test_arr[idx][1] = y
#     idx += 1

data = pd.DataFrame(data_arr,columns = cols_idx)
data.to_csv(save_path+"/data.csv")

# eval_data  = pd.DataFrame(eval_arr,columns = cols_idx)
# eval_data.to_csv(save_path + "/eval_data.csv")

# test_data  =  pd.DataFrame(test_arr,columns = cols_idx)
# test_data.to_csv(save_path+"/test_data.csv")

