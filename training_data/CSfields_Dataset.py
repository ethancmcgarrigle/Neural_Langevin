import numpy as np
import yaml
import torch
from torch.utils.data import Dataset, DataLoader  


## TODO: finalize  
class CSfields_Dataset(Dataset): 

    def __init__(self, simulation_data_phi, simulation_data_phistar, simulation_metadata): 
      """ 
      Args:  
           simulation_data_phi: A 5D np.array containing the simulation CSfield phi data for a given species in k, omega representation. Each element of the list is a 5D np.array of dim: (N_space, Ntau, num_snapshots, N_species, N_simulations).
                    The length of the list is the number of unique simulations, which each generate (50) samples.  
           simulation_data_phistar: "   " for phistar field

           simulation_metadata = a dictionary containing the physical system conditions corresponding to each snapshot. 
                  These are required for calculating an appropriate latent dynamics  

      """
      # dimensionality of receiving CSfields:  num_species x num_snapshots x (n_spatial x ntau) 
      self.phi_field_snapshots = simulation_data_phi
      self.phistar_field_snapshots = simulation_data_phistar
      self.metadata = simulation_metadata
      #self.metadata = simulation_metadata.reindex(simulation_metadata.index.repeat(len(num_simulations[0])))


    def __len__(self):
      return len(self.phi_field_snapshots) # number of simulations x number of snapshots . TODO: do we need to multiply by num_snapshots?   


    #def __getitem__(self, simulation_indx, snapshot_indx):
    def __getitem__(self, idx):
      # unpack various things to get the snapshot from a given simulation for a given species. 
      # "stack the phi and phistar fields by row. 
      #   e.g. CSfields = [ phi(r, tau) ; phistar(r, tau) ] where the ";" means start a new row ala Matlab 
      #phi_field = torch.from_numpy(self.phi_field_snapshots[simulation_indx, 0, snapshot_indx, :, :] )  # simulation indx, species indx, snapshot indx, 
      #phistar_field = torch.from_numpy(self.phistar_field_snapshots[simulation_indx, 0, snapshot_indx, :, :] )  # simulation indx, species indx, snapshot indx, 
      #phi_field = self.phi_field_snapshots[simulation_indx, 0, snapshot_indx, :, :]  # simulation indx, species indx, snapshot indx, 
      #phistar_field = self.phistar_field_snapshots[simulation_indx, 0, snapshot_indx, :, :]   # simulation indx, species indx, snapshot indx, 
      phi_field = self.phi_field_snapshots[idx]  # simulation indx, species indx, snapshot indx, 
      phistar_field = self.phistar_field_snapshots[idx]   # simulation indx, species indx, snapshot indx, 
      # Regularize the data by the max modulus and put into a CSfields tensor object  
      CSfields_tensor = np.row_stack([phi_field/np.max(np.abs(phi_field)), phistar_field/np.max(np.abs(phistar_field))]) # stack the fields into 1 numpy array  
      CSfields_tensor = torch.from_numpy(CSfields_tensor)  # convert to tensor 

      # Complex-valued arithemtic requires complex64 data types 
      CSfields_tensor = CSfields_tensor.to(torch.complex64)

      # Convert meta data to a pytorch tensor 
      sim_params = self.metadata[idx % 50] # a dictionary  
      #sim_params = self.metadata[simulation_indx] # a dictionary  
      metadata_tensor = torch.tensor([val for key, val in sorted(sim_params.items()) ], dtype=torch.float)
      return CSfields_tensor, metadata_tensor 



