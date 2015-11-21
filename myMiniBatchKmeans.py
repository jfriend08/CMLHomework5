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
    self.randState = kwargs.get('randState', 1985)

  def getClosestDist(self, centers, x, returnIdx=False):
    min_dis = sys.float_info.max
    shortestCenter_idx = -1
    for eachC_idx in xrange(len(centers)):
      eachC = centers[eachC_idx]
      if len(eachC) == 0:
        pass
      else:
        dist = euclidean(eachC, x)
        if dist < min_dis:
          min_dis = dist
          shortestCenter_idx = eachC_idx
    return (min_dis if not returnIdx else shortestCenter_idx)

  def getEuclideanDist(self, centers, X):
    return map(lambda eachx:self.getClosestDist(centers, eachx), X)

  def getCloestCenter(self, centers, X):
    return map(lambda eachx:self.getClosestDist(centers, eachx, True), X)

  def _kpp(self, X, n_clusters, **kwargs):
    n_samples, n_features = X.shape
    centers = np.empty((n_clusters, n_features))

    rng = np.random.RandomState(self.randState)
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

  def initCentroids(self, X, initMethod, n_clusters, **kwargs):
    rng = np.random.RandomState(self.randState)
    n_samples = X.shape[0]
    init_size = kwargs.get('init_size', 3*n_clusters)
    init_size = (n_samples if init_size>n_samples else init_size)
    init_indices = rng.random_integers(0, n_samples - 1, init_size)
    X = X[init_indices]
    n_samples = X.shape[0]

    if isinstance(initMethod, str) and initMethod == 'k-means++':
      centers = self._kpp(X, n_clusters)
    elif isinstance(initMethod, str) and initMethod == 'random':
      permuteIdx = rng.permutation(n_samples)[:k]
      centers = X[permuteIdx]

    return centers

  def minibatchCenterUpdate(self, X, centroids):
    C = self.getCloestCenter(centroids, X)




  def _mini_batch_step(self, X, centers, counts, **kwargs):
    compute_squared_diff = kwargs.get('compute_squared_diff', False)
    x2cloestC = np.array(self.getCloestCenter(centers, X))
    k = centers.shape[0]
    squared_diff = 0.0

    for center_idx in range(k):
      center_mask = x2cloestC == center_idx
      count = center_mask.sum()
      if count > 0:
        if compute_squared_diff:
          old_centers=[]
          old_centers = np.copy(centers[center_idx])
        #this center times original counts
        centers[center_idx] *= counts[center_idx]
        #this center plus all x that close to it
        centers[center_idx] += np.sum(X[center_mask], axis=0)
        #update counts
        counts[center_idx] += count
        #this center divided by new counts
        centers[center_idx] /= counts[center_idx]
        if compute_squared_diff:
          diff = centers[center_idx].ravel() - old_centers.ravel()
          squared_diff += np.dot(diff, diff)
    return squared_diff

  def run(self, X):
    n_samples, n_features = X.shape
    counts = np.zeros(self.n_clusters, dtype=np.int32)
    rng = np.random.RandomState(self.randState)
    distances = np.zeros(self.batch_size, dtype=np.float64)
    n_batches = int(np.ceil(float(n_samples) / self.batch_size))
    n_iter = int(self.max_iter * n_batches)
    print "n_iter is", n_iter

    shuffle_indices = rng.random_integers(0, n_samples - 1, n_samples)
    X = X[shuffle_indices] #shuffle samples

    centroids = self.initCentroids(X, self.initMethod, self.n_clusters)
    squared_diff = self._mini_batch_step(X, centroids, counts, compute_squared_diff=True) #list is iterable, will be updated inside _mini_batch_step

    for iteration_idx in range(n_iter):
      print "squared_diff", squared_diff
      minibatch_indices = rng.random_integers(0, n_samples - 1, self.batch_size)
      squared_diff = self._mini_batch_step(X[minibatch_indices], centroids, counts, compute_squared_diff=True)


X, y = pt.dataset_fixed_cov(300, 2, 3) #n, dim, overlapped dist
# pt.plotPCA(X, y)
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
X = min_max_scaler.fit_transform(X)
rng = np.random.RandomState(19850920)
permutation = rng.permutation(len(X))
X, y = X[permutation], y[permutation]
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.1, random_state=2010)

mbk = miniBatchKmeans(8, max_iter=600, batch_size=50)
mbk.run(train_X)






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