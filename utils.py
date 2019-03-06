
import GPy
from GPy.examples.dimensionality_reduction import *
from GPy.examples.classification import *
from GPy.kern import *
from GPy.models import BayesianGPLVM

from sklearn.neighbors import NearestNeighbors

import pods
from pods.datasets import swiss_roll_generated
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import numpy as np

np.random.seed(123344)

def swiss_roll_generator(num_samples=1000, sigma=0.1):
  
  t = 3*np.pi/2*(1+2*np.random.rand(num_samples))
  h = 30*np.random.rand(num_samples)
  y = np.stack([t*np.cos(t),t*np.sin(t),h],axis=1) + sigma*np.random.rand(num_samples,3)
  lab = (t//2 + h//12 + 2) % 2
  return y,lab,t,h
  
  
def classification_error(X,labels):
  
  N = X.shape[0]
  one_NN = NearestNeighbors(n_neighbors=2, algorithm='auto').fit(X)
  _, indices = one_NN.kneighbors(X)
  compt = 0
  for i in range(N):
    if labels[indices[i,0]]!=labels[indices[i,1]]:
      compt+=1

  return compt/N
  
  

def truthworthiness(X,Y,k=5):
  
  N = X.shape[0]
  
  X_one_NN = NearestNeighbors(n_neighbors= N, algorithm='auto').fit(X)
  Y_one_NN = NearestNeighbors(n_neighbors=N, algorithm='auto').fit(Y)
  _, X_ind = X_one_NN.kneighbors(X)
  _, Y_ind = Y_one_NN.kneighbors(Y)
  
  T = 0
  for i in range(N):
    U = np.setdiff1d(X_ind[i,1:k+1],Y_ind[i,1:k+1])
    for j in range(U.shape[0]):
      T = T + (np.where(Y_ind[i,] == U[j])[0] - k)
  T = 1 - 2/(N*k*(2*N - 3*k -1))*T
  
  # Continuity
  C = 0
  for i in range(N):
    V = np.setdiff1d(Y_ind[i,1:k+1], X_ind[i,1:k+1])
    for j in range(V.shape[0]):
      C = C + (np.where(X_ind[i,] == V[j])[0] - k)
  C = 1 - 2/(N*k*(2*N - 3*k -1))*C
  
  return T[0],C[0]
