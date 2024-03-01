import numpy as np
import yaml
import torch

## Helper function to import the CS data files and convert to a tensor format for use in pytorch ## 
def import_CSfield_data(CSfield_fname, input_filename):
  # Return a numpy data object with all rows and cols  
  #data = np.loadtxt(CSfield_fname, unpack=True)
  data = np.loadtxt(CSfield_fname)
  #data = np.transpose(data)
  
  # Get the grid  
  with open(input_filename) as infile:
    params = yaml.load(infile, Loader=yaml.FullLoader)

  # rows = grid pts; cols = vals
  # Data will be input in k-space. Get # of grid pts  
  d = params['system']['Dim']
  Nx = params['simulation']['Nx'] 
  ntau = params['system']['ntau'] 
  #print(data[20, :])
  #print(len(data[0,2*d:]))
  #print(4*ntau)
  assert(4*ntau == len(data[0,2*d:])) # cols w. data = Ntau * 2 (real+imaginary parts) * 2 (for phi & phi^*)  
  num_cols_data = 4*ntau

  N_spatial = Nx
  if(d > 1): 
    Ny = params['simulation']['Ny']
    N_spatial *= Ny
    if(d > 2): 
      Nz = params['simulation']['Nz']
      N_spatial *= Nz

  assert(d == 2) 
  kx = np.unique(data[0,0:N_spatial]) # all the k-values 
  if(d > 1): 
    ky = np.unique(data[1,0:N_spatial])
    if(d > 2): 
      kz = np.unique(data[2,0:N_spatial])

  # Get the number of samples 
  N_samples = int(len(data[:,0])/(N_spatial))
  #print(N_samples)

  # Samples are in the following shape, w = Matsubara freq. and k = wavevector  
  #      w_0(Re, Im)  w_1(Re,Im)  w_2 ...  w_3  w_4  
  # k0 ...
  # k1 ...
  # k2 ... 

  phi_field = np.zeros((N_spatial*N_samples, ntau), dtype=np.complex_)
  phistar_field = np.zeros((N_spatial*N_samples, ntau), dtype=np.complex_)

  # Compile the imaginary time/freq. data:
  for w in range(0, ntau):
    phi_field[:, w] = data[:, 2*d + 2*w] + 1j*data[:, 2*d + 2*w + 1]
    #phi_field.append(phi_w) 
  for w in range(0, ntau):
    phistar_field[:, w]  = data[:, 2*d + 2*w + 2*ntau] + 1j*data[:, 2*d + 2*w + 1 + 2*ntau]

  # Split samples so that now these np arrays should now be (Nx**d, ntau) 
  phi_field = np.split(phi_field, N_samples)
  phistar_field = np.split(phistar_field, N_samples)

  # Now form a 3D numpy array for ease of manipulation and eventual conversion to pytorch tensor  
  #phi_field = np.stack(phi_field)
  #phistar_field = np.stack(phistar_field)

  #phi_field = torch.from_numpy(phi_field)
  #phistar_field = torch.from_numpy(phistar_field)

  return phi_field, phistar_field ## list of 2D numpy arrays, 




def getTimeStamps(input_filename, num_snapshots):
  # This function ouputs a vector of times corresponding to when the snapshots
  #   were taken.  

  # Get the simulation timestepping info 
  with open(input_filename) as infile:
    params = yaml.load(infile, Loader=yaml.FullLoader)

  dt = params['simulation']['dt'] 
  iofreq = params['simulation']['iointerval'] 
  num_warmup_steps = params['simulation']['N_Warm-up_Steps']

  warmup_time = num_warmup_steps * dt
  # report timepts in raw Langevin time 
  timepts = np.zeros(num_snapshots)

  t_0 = 0.
  t_start_recording = warmup_time
  CL_time_interval = dt * iofreq

  timepts += CL_time_interval
  timepts *= np.array(range(0, num_snapshots))
  timepts += warmup_time
  timepts = np.append(np.array([0.]), timepts[0:-1])
  # vector output should be:  timepts = [0, 2000, 2100, 2200, ... ] for 2000 warmup time and 2000 iointerval w. dt = 0.05. 
     # first element is zero because we record the initial condition and then wait for a warmup period.
  return timepts 


