import numpy as np
import random, sys
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA

def lossHinge(yt):
    return max(0, 1-yt)

def lossHuberHinge (yt):
  h = 0.3
  if yt > 1+h:
    return 0
  elif yt < 1-h:
    return 1 - yt
  else:
    return ((1+h-yt)**2)/(4*h)

def getAccuracyOverIteration(wIter, X, y):
  def getAccuracy(w, X, y):
    predict = np.apply_along_axis(lambda eachx: np.dot(w,eachx), 1, X)
    diff = predict*y
    return len(np.where(diff>0)[0])/float(len(X))
  accuracy = map(lambda w: getAccuracy(w, X, y), wIter)
  return accuracy


# generate datasets
def dataset_fixed_cov(n, dim, C):
  '''Generate 2 Gaussians samples with the same covariance matrix'''
  # n, dim = 300, 2
  np.random.seed(0)
  # C = np.array([[0., -0.23], [0.83, .23]])
  X = np.r_[np.dot(np.random.randn(n, dim), C),
            np.dot(np.random.randn(n, dim), C) + np.array([1, 1])]
  y = np.hstack((np.zeros(n), np.ones(n)))
  return X, y

def dataset_cov(n, dim, C):
  '''Generate 2 Gaussians samples with different covariance matrices'''
  # n, dim = 300, 2
  np.random.seed(0)
  # C = np.array([[0., -1.], [2.5, .7]]) * 2.
  X = np.r_[np.dot(np.random.randn(n, dim), C),
            np.dot(np.random.randn(n, dim), C.T) + np.array([1, 4])]
  y = np.hstack((np.zeros(n), np.ones(n)))
  return X, y

class gradientDescent(object):
  def __init__(self, X, y, **kwargs):
    self.X = X
    self.y = y
    self.h = kwargs.get('h', 0.3)
    self.c = kwargs.get('c', 1)
    self.maxiter = kwargs.get('maxiter', 1000)
    self.ita = kwargs.get('ita', 0.11)

  def computeYT(self, w):
    wt = map(lambda x: np.dot(w, x), self.X)
    return np.dot(self.y, wt)

  def computeYderT(self):
    n = float(len(self.X[0]))
    derW = np.ones(n)
    derwt = map(lambda x: np.dot(derW, x), self.X)
    return np.dot(self.y, derwt)

  def lossF(self, myinput):
    y, wx = myinput
    if y*wx > 1+self.h:
      return 0
    elif y*wx < 1-self.h:
      return 1-y*wx
    else:
      return (1+self.h-y*wx)**2/(4*self.h)

  def lossF_dir(self, w, x, myinput):
    y, wx, dirwx = myinput
    # print "y", y, "wx", wx, "dirwx", dirwx
    unit = np.ones(self.X.shape[1])
    # dirwx = map(lambda x: np.dot(unit, x), self.X)
    if y*wx > 1+self.h:
      # print "case 1"
      return 0
    elif y*wx < 1-self.h:
      # print "case 2"
      return -y*dirwx
    else:
      # print "case 3"
      return (2*(1+self.h-y*wx)/(4*self.h))*(-1)*y*dirwx

  def compute_obj(self, w):
    n = self.X.shape[0]
    wx = np.apply_along_axis(lambda x: np.dot(w, x), 1, self.X)
    return np.dot(w, w) + (self.c/n)*sum(np.apply_along_axis(self.lossF, 1, zip(self.y, wx)) )

  def compute_grad(self, w):
    unit = np.ones(self.X.shape[1])
    wx = map(lambda x: np.dot(w, x), self.X)
    dirwx = map(lambda x: np.dot(unit, x), self.X)
    result = map(lambda x: self.lossF_dir(w, self.X, x), zip(self.y, wx, dirwx))
    # print "sum(result)", sum(result)
    return 2*w + sum(result)

    # unit = np.ones(self.X.shape[1])
    # wx = np.apply_along_axis(lambda x: np.dot(unit, x), 1, self.X)
    # return 2*w + sum(map(self.lossF, zip(self.y, wx)) )

  def getNumericalResultAtEachDirection(self, compute_obj, w, epslon, eachdir):
    return (compute_obj(w+epslon*eachdir) - compute_obj(w-epslon*eachdir))/(2*epslon)

  def grad_checker(self, compute_obj, compute_grad, w):
    epslon = float(0.1/10**8)
    uniDirection = np.zeros((len(w), len(w)), int)
    np.fill_diagonal(uniDirection, 1)
    numericalResult = np.apply_along_axis(lambda x: self.getNumericalResultAtEachDirection(compute_obj, w, epslon, x), 1, uniDirection)
    analyticResult = compute_grad(w)
    print "numericalResult", numericalResult
    print "analyticResult", analyticResult
    return sum(numericalResult-analyticResult)/sum(analyticResult)

  def my_gradient_decent(self, w, **kwargs):
    self.h = kwargs.get('h', 0.3)
    self.c = kwargs.get('c', 1)
    self.maxiter = kwargs.get('maxiter', 100)
    self.ita = kwargs.get('ita', 0.11)
    Step_backtrack = kwargs.get('Step_backtrack', False)
    compute_obj = kwargs.get('compute_obj', self.compute_obj) #user can specify the function
    compute_grad = kwargs.get('compute_grad', self.compute_grad) #user can specify the function
    iterCount = 0
    wIter = [w]
    def getStep_backtrack(w):
      if Step_backtrack:
        direction = compute_grad(w)
        t = 1
        a = 0.9
        b = 0.5
        dw = np.ones(len(w))
        while compute_obj(w+t*dw) > compute_obj(w) + a*t*np.dot(compute_grad(w),dw):
          t *= b
        return t

    print "------------------", "h", self.h, "c", self.c, "maxiter", self.maxiter, "ita", self.ita, "------------------"
    while iterCount < self.maxiter:
      if iterCount == 1 and Step_backtrack:
        self.ita = getStep_backtrack(w)
      w = w - self.ita*compute_grad(w)
      wIter.append(w)
      iterCount+=1
    return wIter, w, iterCount



''' generate Gaussian distributions with fix covariance'''
# X, y = dataset_fixed_cov(300, 2, C = np.array([[0.5, -0.23], [0.83, .23]]) ) #n, d, C
# pca = PCA()
# pca.fit(X)
# X_pca = pca.transform(X)
# plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, linewidths=0, s=30)
# plt.show()

import points as pt
from sklearn.cross_validation import train_test_split
X, y = pt.dataset_fixed_cov(500, 10, 3) #n, dim, overlapped dist
print "X.shape", X.shape
min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
X = min_max_scaler.fit_transform(X)
rng = np.random.RandomState(19850920)
permutation = rng.permutation(len(X))
X, y = X[permutation], y[permutation]
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.1, random_state=2010)
# pt.plotPCA(X, y)

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
X = min_max_scaler.fit_transform(X)
rng = np.random.RandomState(19850920)
permutation = rng.permutation(len(X))
X, y = X[permutation], y[permutation]
train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.1, random_state=2010)

n, dim = X.shape
w = np.zeros(dim)
mygd = gradientDescent(train_X, train_y)
# mygd.my_gradient_decent(w)
wIter, w, iterCount = mygd.my_gradient_decent(w, maxiter=50, ita=0.1, c=1, Step_backtrack=True)
'''----------------------------------------------'''


# w = np.array([1, 1, 0, 1, 0])
# mygd.compute_obj(w)
# mygd.compute_grad(w)

# '''checking correctness'''
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
# x = np.array([np.zeros(5), np.zeros(5), np.ones(5), np.ones(5), np.zeros(5), np.zeros(5)])
# x = min_max_scaler.fit_transform(x)
# y = np.array([-1, -1, 1, 1, -1, -1])

# mygd = gradientDescent(x, y)
# w = np.array([1, 1, 0, 1, 0])
# mygd.compute_obj(w)
# mygd.compute_grad(w)

# for i in xrange(5):
#   w = np.random.randint(100, size=(1, 5)).flatten()
#   print i, "th checking: error sum\n", mygd.grad_checker(mygd.compute_obj, mygd.compute_grad, w)
#   print "----------------------"