import numpy 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import DataLoader, random_split
from torch.autograd import Variable
from reduced_model.autoEncoder import *
from reduced_model.mmd_loss import  MMD_loss 
from reduced_model.training import *
from training_data.CSfields_Dataset import *
import scipy.io
from timeit import default_timer
import torch.optim as optim
import h5py
import pdb 
import os 
import sys
import json

''' Main script for training the autoencoder neural network '''


# Set the device to GPU or CPU 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = 'cpu'
# set rand seed 
torch.manual_seed(0)

#### Import training data ### 
# Load the data and then create custom data set object 
training_data_directory = './training_data/data/'
print('Loading the training data')
with h5py.File(training_data_directory + 'hmg_training_data.h5', 'r') as f:
  all_phi_fields = f['phi_fields_training'][:] 
  all_phistar_fields = f['phistar_fields_training'][:] 
  # Load the metadata 
  #simulation_metadata = {key: f.attrs[key] for key in f.attrs} 
  simulation_metadata = {int(idx): json.loads(f.attrs[idx]) for idx in f.attrs}

# Double check the loading was done correctly) 
#print(all_phi_fields[0])
#print(all_phistar_fields[0])
#print(simulation_metadata[0])
#sys.exit()

print('Creating the torch data set')
Simulation_Dataset = CSfields_Dataset(all_phi_fields, all_phistar_fields, simulation_metadata)


#sys.exit()
# Separate the data into training and testing data sets & Get the custom data loaders
pcnt_train = 0.8 # percentage of data devoted to training  
pcnt_test = np.round(1.0 - pcnt_train, 4)   # percentage of data devoted to testing  
num_training_samples = int(pcnt_train * len(Simulation_Dataset))
num_testing_samples = int(pcnt_test * len(Simulation_Dataset))

training_dataset, testing_dataset = random_split(Simulation_Dataset, [num_training_samples, num_testing_samples])

_batchSize = 20
# Create training and testing data-loaders 
# use "pin_memory" arg to create cuda tensors? 
training_loader = DataLoader(training_dataset, batch_size=_batchSize, shuffle=True, pin_memory=False)
testing_loader = DataLoader(testing_dataset, batch_size=_batchSize, shuffle=False, drop_last=True, pin_memory=False)


#train_loader, test_loader = dataloader_neuralCL(Simulation_DataSet, ntrain=num_training_samples, 
                                                 #ntest=num_testing_samples, batch_size=10) 

num_layers = 3
nodes_per_layer = np.array([5000, 1000, 100])

# Define our model
N_spatial = 20 * 20 
ntau = 64 
Autoencoder_model = AutoEncoder(2*N_spatial*ntau, nodes_per_layer, num_layers, 16, 'ReLU')
#Autoencoder_model = AutoEncoder(50*50, np.array([100, 100]), 2, 50, 'ReLU')

print('Number of parameters in the model: ' + "{:e}".format(count_parameters(Autoencoder_model)))


# Define the loss functions and train the model 
loss = MMD_loss()
#loss = nn.MSELoss(reduction='mean')

_epochs = 50
_learning_rate = 0.025
_training_io_interval = 5
_isTimingTraining = True
_isTimingEvaluation = True
model, training_loss, testing_loss, time_train, time_eval = train_neuralCL(Autoencoder_model, training_loader, testing_loader, device, loss, 
                                                                            _batchSize, _epochs, _learning_rate, _training_io_interval, 
                                                                            _isTimingTraining, _isTimingEvaluation)

#sys.exit()

# Save the trained model 
torch.save(model.state_dict(), 'model_state_dict.pth') # recommended to only save the state dictionary 

np.savetxt('training_testing_data.dat', np.column_stack([np.array(training_loss).real, np.array(training_loss).imag, np.array(testing_loss).real, np.array(testing_loss).imag]), header = 'Re_training_loss Im_training_Loss Re_testing_loss Im_testing_loss')

# To load: #
# model2 = Autoencoder_model(...) 
#model2.load_state_dict(torch.load('model_state_dict.pth'))






