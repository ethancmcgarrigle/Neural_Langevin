import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import operator
from timeit import default_timer
from functools import reduce
import sys 
#from reduced_model.mmd_loss import * 

def train_neuralCL(model, train_loader, test_loader, device, loss_fxn, batch_size, epochs, learning_rate, training_io_interval, time_train, time_eval):  

    optimizer = optim.Adam(model.parameters(), lr = learning_rate, weight_decay = 1E-4) 

    ntrain = len(train_loader.dataset)
    ntest = len(test_loader.dataset)

    model.to(device)

    losses_train = []
    losses_test = []

    times_train = [] 
    times_eval = []

    try:
      for ep in range(epochs):
        # Train the model 
        model.train()
        train_loss = 0.

        # Compute loss by loop over training data 
        for _batch, _batch_metadata in train_loader:

          loss = 0.
          #print(_batch.shape)
          #sys.exit() 
          #print(sample_dict)

          t1 = default_timer()
          _batch = _batch.reshape(batch_size, -1)
          #sample_ = sample_.flatten()
          # move the batch and the model to GPU 
          _batch = _batch.to(device)
          prediction = model(_batch)

          # Compute the loss  
          loss = loss_fxn(prediction.reshape(batch_size, -1), _batch.reshape(batch_size, -1))
          print(loss.item())

          train_loss += loss.item()

          # Backpropagate to get dL/dtheta where theta are the model parameters 
          loss.backward()
          # Take a learning step 
          optimizer.step()
          # Reset gradients and loss 
          optimizer.zero_grad()

          times_train.append(default_timer()-t1)

        test_loss = 0.
        with torch.no_grad(): # don't evaluate gradients 
          for _batch, _batch_metadata in test_loader:
            loss = 0.
            #print(_batch.shape)
            _batch = _batch.reshape(batch_size, -1)
            #print(_batch.shape)
            _batch = _batch.to(device)

            t1 = default_timer()
            prediction = model(_batch)
            times_eval.append(default_timer()-t1)

            # Compute the loss  
            loss = loss_fxn(prediction.reshape(batch_size, -1), _batch.reshape(batch_size, -1))

            test_loss += loss.item()

        # Check to see if we've plateau'd and should stop             
 #        if plateau_patience is None:
 #          scheduler.step()
 #        else:
 #          scheduler.step(test_loss/ntest)
 #        if plateau_terminate is not None:
 #          early_stopping(test_loss/ntest, model)
 #          if early_stopping.early_stop:
 #            print("Early stopping")
 #            break
        
        # Print out losses at the iointerval 
        if ep % training_io_interval == 0:
          losses_train.append(train_loss/ntrain)
          losses_test.append(test_loss/ntest)
          print('Epoch {:04d} | Total Train Loss {:.6f} | Total Test Loss {:.6f}'.format(ep, train_loss / ntrain, test_loss / ntest))

      if time_train and time_eval:
        return model, losses_train, losses_test, times_train, times_eval 
      elif time_train and not time_eval:
        return model, losses_train, losses_test, times_train
      elif time_eval and not time_train:
        return model, losses_train, losses_test, times_eval 
      else:
        return model, losses_train, losses_test
        
    except KeyboardInterrupt:
      if time_train and time_eval:
        return model, losses_train, losses_test, times_train, times_eval 
      elif time_train and not time_eval:
        return model, losses_train, losses_test, times_train
      elif time_eval and not time_train:
        return model, losses_train, losses_test, times_eval 
      else:
        return model, losses_train, losses_test






def count_parameters(model):
  # Counts the number of parameters being optimized in the NN model 
  num_params = 0
  for p in list(model.parameters()):
    num_params += reduce(operator.mul, list(p.size()))

  return num_params

