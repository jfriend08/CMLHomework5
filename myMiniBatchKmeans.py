import numpy as np
import random, sys
import points as pt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from  scipy.spatial.distance import euclidean

def dataset_fixed_cov(n, dim, C):
  '''Generate 2 Gaussians samples with the same covariance matrix'''
  # n, dim = 300, 2
  np.random.seed(0)
  # C = np.array([[0., -0.23], [0.83, .23]])
  X = np.r_[np.dot(np.random.randn(n, dim), C),
            np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
  y = np.hstack((np.zeros(n), np.ones(n)))
  return X, y

class miniBatchKmeans(object):
  def __init__(self, n_clusters=8, initMethod='k-means++', **kwargs):
    self.n_clusters = n_clusters
    self.initMethod = initMethod
    self.max_iter = kwargs.get('max_iter', 100)
    self.batch_size = kwargs.get('batch_size', 100)

  def getClosestDist(self, centers, x):
    min_dis = sys.float_info.max
    shortestCenter_idx = -1
    for eachC in centers:
      if len(eachC) == 0:
        pass
      else:
        dist = euclidean(eachC, x)
        if dist < min_dis:
          min_dis = dist
    return min_dis

  def getEuclideanDist(self, centers, X):
    return map(lambda eachx:self.getClosestDist(centers, eachx), X)

  def _kpp(self, X, n_clusters, randState, **kwargs):
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features))

    rng = np.random.RandomState(randState)
    first_id = rng.randint(n_samples)
    centers[0] = X[first_id]

    distances = self.getEuclideanDist(centers, X)
    probabilityDist = np.array(distances)/sum(distances)

    for cidx in xrange(1, n_clusters):
      idx = np.random.choice(len(X), 1, p=probabilityDist)
      centers[cidx] = X[idx]
      distances = self.getEuclideanDist(centers, X)
      probabilityDist = np.array(distances)/sum(distances)
    return centers

  def initCentroids(self, X, initMethod, n_clusters, randState, **kwargs):
    rng = np.random.RandomState(randState)
    n_samples = X.shape[0]
    init_size = kwargs.get('init_size', 3*n_clusters)
    init_size = (n_samples if init_size>n_samples else init_size)
    init_indices = rng.random_integers(0, n_samples - 1, init_size)
    X = X[init_indices]
    n_samples = X.shape[0]

    if isinstance(initMethod, str) and initMethod == 'k-means++':
      centers = self._kpp(X, n_clusters, randState)
    elif isinstance(initMethod, str) and initMethod == 'random':
      permuteIdx = rng.permutation(n_samples)[:k]
      centers = X[permuteIdx]

    return centers

  def run(self, X):
    n_samples, n_features = X.shape
    distances = np.zeros(self.batch_size, dtype=np.float64)
    n_batches = int(np.ceil(float(n_samples) / self.batch_size))
    n_iter = int(self.max_iter * n_batches)
    centroids = self.initCentroids(X, self.initMethod, self.n_clusters, 19850920)
    # for iteration_idx in range(n_iter):


X, y = pt.dataset_fixed_cov(300, 10, 3) #n, dim, overlapped dist
print "X.shape", X.shape, "X[0]", X[0]
pt.plotPCA(X, y)
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
# X = min_max_scaler.fit_transform(X)
# rng = np.random.RandomState(19850920)
# permutation = rng.permutation(len(X))
# X, y = X[permutation], y[permutation]
# train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.1, random_state=2010)

# mbk = miniBatchKmeans(8, max_iter=10, batch_size=50)
# mbk.run(train_X)


# batch_size = 10
# n_samples = 20
# max_iter = 10
# init_size = None
# n_clusters = 3

# distances = np.zeros(batch_size, dtype=np.float64)
# n_batches = int(np.ceil(float(n_samples) / batch_size))
# n_iter = int(max_iter * n_batches)

# print "distances.shape", distances.shape, "distances", distances
# print "n_batches", n_batches
# print "n_iter", n_iter

# if init_size is None:
#   init_size = 3 * batch_size
# if init_size > n_samples:
#   init_size = n_samples
# init_size_ = init_size

# print "init_size", init_size
# rng = np.random.RandomState(19850920)
# validation_indices = rng.random_integers(0, n_samples - 1, init_size)
# print "validation_indices", validation_indices
# minibatch_indices = rng.random_integers( 0, n_samples - 1, batch_size)
# print "minibatch_indices", minibatch_indices
# # X_valid = X[validation_indices]
# # x_squared_norms_valid = x_squared_norms[validation_indices]
# counts = np.zeros(n_clusters, dtype=np.int32)
# print "counts", counts