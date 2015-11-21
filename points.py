from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.decomposition import PCA


def dataset_fixed_cov(n,dim):
  '''Generate 2 Gaussians samples with the same covariance matrix'''
  np.random.seed(0)
  w = np.random.randint(100, size=(1, 5)).flatten()
  C = np.random.normal(0, 1, (dim, dim))
  print C
  print "C.shape", C.shape
  X = np.r_[np.dot(np.random.randn(n, dim), C),
            np.dot(np.random.randn(n, dim), C) + 4*np.ones(dim)]
  y = np.hstack((np.zeros(n), np.ones(n)))
  return X, y

def plotPCA(X, y):
  pca = PCA()
  pca.fit(X)
  X_pca = pca.transform(X)
  plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, linewidths=0, s=30)
  plt.show()