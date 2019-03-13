import numpy as np

def __init__(self, state_dim, obs_dim):
		self.state_dim = state_dim
		self.obs_dim   = obs_dim
		
        self.Q 		 = np.matrix( np.eye(state_dim)*1e-4 )			  # Process noise
		self.R		 = np.matrix( np.eye(obs_dim)*0.01 )			  # Observation noise
		self.A		 = np.matrix( np.eye(state_dim) )			  # Transition matrix
		self.H		 = np.matrix( np.zeros((obs_dim, state_dim)) )		  # Measurement matrix
		self.K		 = np.matrix( np.zeros_like(self.H.T) )			  # Gain matrix
		self.P		 = np.matrix( np.zeros_like(self.A) )			  # State covariance
		self.x		 = np.matrix( np.zeros((state_dim, 1)) )		  # The actual state of the system
	