import numpy as np
from data_formatting_tools import *
import torch 
import os 
import h5py
import json
import os 
import subprocess

# Import training data from all these simulations  
beta = np.array([1.0, 0.25,  0.05, 0.025])
g = np.linspace(0.1, 0.5, 5)
mu = np.linspace(0.5, 2.0, 5)

beta = beta[0:2]
#g = g[0:3]
#mu = mu[0:3]

g = np.round(g, 3)
mu = np.round(mu, 3)
beta = np.round(beta, 3)


num_snapshots = 50
num_species = 1
ctr = 0

simulation_parameters = {} # dictionary for system physical parameters 
all_phi_fields = [] # num_simulations x num_species x num_snapshots x (n_spatial x ntau) 
all_phistar_fields = [] # num_simulations x num_species x num_snapshots x (n_spatial x ntau) 

# Clean the training data from a previous run  
print('Cleaning old training data')
subprocess.call('rm ./data/*.h5', shell=True)

initial_dir = './data'
os.chdir(initial_dir)

for beta_ in beta:
  # directory name 
  #print(os.getcwd())
  outer_dir_name = './' + 'beta_' + str(beta_)
  # change directory
  os.chdir(outer_dir_name)

  for g_ in g:
    # directory name 
    inner_dir_name = 'g_' + str(g_) 
    # change directory
    os.chdir(inner_dir_name)

    for mu_ in mu:
      # directory name 
      inner_dir_name = 'mu_' + str(mu_) 
      # change directory
      os.chdir(inner_dir_name)

      print(os.getcwd())
      # Run the import data script to parse through the data  
      fname_prefix = 'phi_phistar_training'
      input_fname = 'input.yml'
      #phi_fields = [] # list of length N_species 
      #phistar_fields = [] # list of length N_species 
 #      for alpha in range(0, num_species):
 #        CS_data_fname = fname_prefix + str(alpha) + '.dat'
      CS_data_fname = fname_prefix + str(0) + '.dat'
 #        # import the data here 
      phi, phistar = import_CSfield_data(CS_data_fname, input_fname) # returns 3D np array (N_samples x (Nspace x ntau))
      #phi_fields.append(phi) 
      #phistar_fields.append(phistar) 
      
      # Generate the corresponding list of CL_timepoints  
      # np.stack converts outer species list structure to 4D numpy array 
      #phi_fields = np.stack(phi_fields)
      #phistar_fields = np.stack(phistar_fields) # num_species x num_snapshots x (n_spatial x ntau) 
      #all_phi_fields.append(phi_fields) # appends to list of simulations  
      #all_phistar_fields.append(phistar_fields) 
      all_phi_fields += phi # combine list with running list for all the snapshots 
      all_phistar_fields += phistar  

      CL_timepoints = getTimeStamps(input_fname, num_snapshots) 
      print(CL_timepoints)

        
      # Open the input file for that simulation, then record the simulation conditions in the dictionary 
      with open(input_fname) as infile:
        params = yaml.load(infile, Loader=yaml.FullLoader)

      simulation_parameters[ctr] = {
           'sim_ID' : ctr,
           'beta': params['system']['beta'],
           'ntau': params['system']['ntau'],
           'dim': params['system']['Dim'],
           'mu' : params['system']['mu'],
           'L'  :  params['system']['CellLength-x'], 
           'g'  :  params['system']['pairinteraction']['u0'],
           'Nx' :  params['simulation']['Nx'], 
           'dt' :  params['simulation']['dt'], 
           'lambda' : 6.0596534037} # hbar^2/2m  
  
      print(simulation_parameters[ctr])
      ctr += 1
  
      os.chdir('../') # leave mu dir

    os.chdir('../') # leave g dir

  os.chdir('../')

# Convert to 5D numpy array (num_sims x num_species x num_snapshots x (nspace x ntau)) since the custom dataset object expects this as input
#all_phi_fields = np.stack(all_phi_fields)
#all_phistar_fields = np.stack(all_phistar_fields)

# Reshape for now
 #num_simulations = len(all_phi_fields[:, 0, 0, 0, 0])
 #num_snapshots = len(all_phi_fields[0, 0, :, 0, 0]) 
 #num_Nx = len(all_phi_fields[0,0,0,:,0])
 #num_Ntau = len(all_phi_fields[0,0,0,0,:])
 #all_phi_fields = all_phi_fields.reshape((1, num_simulations*num_snapshots, num_Nx, num_Ntau))
 #assert(num_Nx == 400)
 #assert(num_Ntau == 64)

# Export data to be loaded later   
#np.save('phi_fields_trainingData.npy', all_phi_fields)
#np.save('phistar_fields_trainingData.npy', all_phistar_fields)
print('Saving data to : ' + str(os.getcwd()))
with h5py.File('hmg_training_data.h5', 'w') as f:
  print('Saving the training data')
  # Saves the list of snapshots. Each element of the list is a snapshot (2D numpy array)
  f.create_dataset('phi_fields_training', data=all_phi_fields)
  f.create_dataset('phistar_fields_training', data=all_phistar_fields)

  # Add metadata 
  for idx, params in simulation_parameters.items():
    params_str = json.dumps(params)
    f.attrs[str(idx)] = params_str
 #  for key, value in simulation_parameters.items():
 #    f.attrs[key] = value 

