import numpy as np

def KF(state_dim, obs_dim):
	state_dim = state_dim
	obs_dim   = obs_dim
	R = np.matrix( np.eye(obs_dim)*0.01 )			  # Observation noise
	A = np.matrix( np.eye(state_dim) )			  # Transition matrix
	H = np.matrix( np.zeros((obs_dim, state_dim)) )		  # Measurement matrix
	K = np.matrix( np.zeros_like(self.H.T) )		  # Gain matrix
	P = np.matrix( np.zeros_like(self.A) )			  # State covariance
	x = np.matrix( np.zeros((state_dim, 1)) )		  # The actual state of the system
	
