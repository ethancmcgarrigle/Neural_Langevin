import numpy 
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.autograd import Variable
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear 
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d

# Complex neural network modules can be found: https://github.com/wavefrontshaping/complexPyTorch # 


class Encoder(nn.Module):
    def __init__(self, input_dimension, num_nodes_per_layer, num_layers, latent_dimension, activation_str):
      super().__init__()
      ''' 
         Encoder neural network class. 
         Inputs: 
             - input_dimension is the number of elements of the vector input 
             - num_nodes_per_layer is a list of the number of nodes in each hidden layer.  
               - This should be a vector of int's of length num_layers 
             - activation_str is a string specifying the activation function. 
      '''
      # Perform safety checks 
      assert len(num_nodes_per_layer) == num_layers, 'Length of nodes vector needs to match num_layers argument'
      # Strictly use for dimensionality reduction # 
      for node_count in num_nodes_per_layer: assert node_count < input_dimension, 'Warning: Encoder is increasing the dimensionality' 
      for node_count in num_nodes_per_layer: assert latent_dimension < node_count, 'Warning: number of nodes in a layer is smaller than the desired latent dimension' 

      print()
      print('Constructing a dense encoder neural network with ' + str(num_layers) + ' layers') 
      print()

      # Parse activation string 
      self.activation_str = activation_str
      if(activation_str == 'ReLU'):
        activation_function = nn.ReLU
      elif (activation_str == 'Tanh'):
        activation_function = nn.Tanh
      else: ## Default to RELU 
        activation_function = nn.ReLU

      hidden_layers = []
      # First layer
      fc_layer1 = nn.Linear(input_dimension, num_nodes_per_layer[0])
      hidden_layers.append(fc_layer1)
      hidden_layers.append(activation_function())

      # Build a sequence of hidden layers 
      for layer_num in range(1, num_layers):
        hidden_layers.append(nn.Linear(num_nodes_per_layer[layer_num-1], num_nodes_per_layer[layer_num]))
        hidden_layers.append(activation_function())

      # Add a linear layer for the final layer 
      fc_final_layer = nn.Linear(num_nodes_per_layer[num_layers-1], latent_dimension)
      hidden_layers.append(fc_final_layer)

      # Finish NN creation with Sequential()
      self.encoder = nn.Sequential(*hidden_layers)
      ## Add additional vars for varitional autoencoder version   
      

    def forward(self, X):
      # Forward pass for encoder class. 
      X = self.encoder(X)
      return X


class Decoder(nn.Module):
    def __init__(self, latent_dimension, num_nodes_per_layer, num_layers, output_dimension, activation_str):
      super().__init__()
      ''' 
         Decoder neural network class. 
         Inputs: 
             - latnet_dimensions is the number of elements of the vector input 
             - output_dimension is the number of elements of the vector output (i.e. after decoding) 
             - num_nodes_per_layer is a list of the number of nodes in each hidden layer.  
               - This should be a vector of int's of length num_layers 
             - activation_str is a string specifying the activation function. 
      '''
      assert len(num_nodes_per_layer) == num_layers, 'Length of nodes vector needs to match num_layers argument'
      # Strictly use for dimensionality reduction # 
      for node_count in num_nodes_per_layer: assert node_count < output_dimension, 'Warning: Decoder is decreasing the dimensionality' 
      for node_count in num_nodes_per_layer: assert latent_dimension < node_count, 'Warning: number of nodes in a layer is smaller than the desired latent dimension' 

      print()
      print('Constructing a dense decoder neural network with ' + str(num_layers) + ' layers') 
      print()

      # Parse activation string 
      self.activation_str = activation_str
      if(activation_str == 'ReLU'):
        activation_function = nn.ReLU
      elif (activation_str == 'Tanh'):
        activation_function = nn.Tanh
      else: ## Default to RELU 
        activation_function = nn.ReLU

      hidden_layers = []
      # First layer
      fc_layer1 = nn.Linear(latent_dimension, num_nodes_per_layer[-1])
      hidden_layers.append(fc_layer1)
      hidden_layers.append(activation_function())

      # Build a sequence of hidden layers 
      for layer_num in reversed(range(1, num_layers)):
        hidden_layers.append(nn.Linear(num_nodes_per_layer[layer_num], num_nodes_per_layer[layer_num-1]))
        hidden_layers.append(activation_function())

      # Add a linear layer for the final layer 
      fc_final_layer = nn.Linear(num_nodes_per_layer[0], output_dimension)
      hidden_layers.append(fc_final_layer)

      # Finish NN creation with Sequential()
      self.decoder = nn.Sequential(*hidden_layers)
      ## Add additional vars for varitional autoencoder version   
      

    def forward(self, X):
      # Forward pass for encoder class. 
      X = self.decoder(X)
      return X



class AutoEncoder(nn.Module):
    '''
       Class definition for a variational autoencoder to be used to encode a latent space
    '''

    def __init__(self, input_dimension, num_nodes_per_layer, num_layers, latent_dimension, activation_str):

      super().__init__()
      print('Creating autoencoder neural network model')
      print('Creating Encoder')
      #self.encoder = nn.Sequential(
 #            nn.Linear(input_dimension, 1000),
 #            nn.ReLU(True),
 #            nn.Linear(1000, 128),
 #            nn.ReLU(True),
 #            nn.Linear(128, 64),
 #            nn.ReLU(True),
 #            nn.Linear(64, 32),
 #            nn.ReLU(True),
 #            nn.Linear(32, 16)
      self.fc1 = ComplexLinear(input_dimension, 100)
      self.fc2 = ComplexLinear(100, 32)
      self.fc3 = ComplexLinear(32, 16)
          
      self.bfc1 = ComplexLinear(16, 32)
      self.bfc2 = ComplexLinear(32, 100)
      self.bfc3 = ComplexLinear(100, input_dimension)
      #self.encoder = Encoder(input_dimension, num_nodes_per_layer, num_layers, latent_dimension, activation_str)
      print('Creating Decoder')
      #self.decoder = Decoder(latent_dimension, num_nodes_per_layer, num_layers, input_dimension, activation_str)
 #      self.decoder = nn.Sequential(
 #            ComplexLinear(16, 32),
 #            complex_relu(True),
 #            ComplexLinear(32, 64),
 #            complex_relu(True),
 #            ComplexLinear(64, 128),
 #            complex_relu(True),
 #            ComplexLinear(128, 1000),
 #            complex_relu(True),
 #            ComplexLinear(1000, input_dimension)
 #            #nn.Sigmoid()
 #      )
 #            nn.Linear(16, 32),
 #            nn.ReLU(True),
 #            nn.Linear(32, 64),
 #            nn.ReLU(True),
 #            nn.Linear(64, 128),
 #            nn.ReLU(True),
 #            nn.Linear(128, 1000),
 #            nn.ReLU(True),
 #            nn.Linear(1000, input_dimension),



    def forward(self, x):
      #print(x.shape)
      #x = x.view(x.size(0), -1)
      #x = x.squeeze(0)
      # Encoder
      #print(x.dtype)
      #x = x.to(torch.complex64)
      #print(x.dtype)
      x = self.fc1(x)
      x = complex_relu(x)
      x = self.fc2(x)
      x = complex_relu(x)
      x = self.fc3(x)
      x = complex_relu(x)

      # Decoder
      x = self.bfc1(x) 
      x = complex_relu(x)
      x = self.bfc2(x) 
      x = complex_relu(x)
      x = self.bfc3(x) 
      #x = complex_relu(x)
      #print(X.dtype) 
      #z = self.encoder(X)
      #print(z.dtype) 
      #z = self.encoder(X)
      #X_hat = self.decoder(z)
      #X_hat = X.view(X.size(0), 1, 800, 64)
      return x # should be the same as X 
