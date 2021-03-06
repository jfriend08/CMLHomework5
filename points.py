from scipy import linalg
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import colors
from sklearn.decomposition import PCA


def dataset_fixed_cov(n,dim, dist):
  '''Generate 2 Gaussians samples with the same covariance matrix'''
  np.random.seed(0)
  w = np.random.randint(100, size=(1, 5)).flatten()
  C = np.random.normal(0, 1, (dim, dim))
  X = np.r_[np.dot(np.random.randn(n, dim), C),
            np.dot(np.random.randn(n, dim), C) + dist*np.ones(dim)]
  y = np.hstack((-1*np.ones(n), np.ones(n)))
  return X, y

def plotPCA(X, y):
  pca = PCA()
  pca.fit(X)
  X_pca = pca.transform(X)
  plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, linewidths=0, s=30)
  plt.show()