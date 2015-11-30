import random, sys
import numpy as np
import gradientDescent as gd
import matplotlib.pyplot as plt
import points as pt
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.decomposition import PCA


class svm(object):
  def __init__(self):
    pass

  def gd(self, X, y, w, compute_objF, compute_gradF, **kwargs):
    print "start gd"
    h = kwargs.get('h', 0.3)
    c = kwargs.get('c', 1)
    maxiter = kwargs.get('maxiter', 100)
    ita = kwargs.get('ita', 0.11)
    Step_backtrack = kwargs.get('Step_backtrack', False)
    stopMethod = kwargs.get('stopMethod', None) #user can specify the function
    mygd = gd.gradientDescent(X, y)
    if compute_objF=="Default" or compute_gradF=="Default":
      return mygd.my_gradient_decent(w, h=h, c=c, maxiter=maxiter, ita=ita, Step_backtrack=Step_backtrack, stopMethod=stopMethod)
    else:
      return mygd.my_gradient_decent(w, compute_obj=compute_objF, compute_grad=compute_gradF, h=h, c=c, maxiter=maxiter, ita=ita, Step_backtrack=Step_backtrack, stopMethod=stopMethod)

  def sgd (self, X, y, w, compute_objF, compute_gradF, **kwargs):
    print "start sgd"
    h = kwargs.get('h', 0.3)
    c = kwargs.get('c', 1)
    maxiter = kwargs.get('maxiter', 5)
    ita = kwargs.get('ita', 0.11)
    Step_backtrack = kwargs.get('Step_backtrack', False)
    stopMethod = kwargs.get('stopMethod', None) #user can specify the function
    mysgd = gd.gradientDescent(X, y)
    itaOverIteration = kwargs.get('itaOverIteration', False)
    tnot = kwargs.get('tnot', 1)
    if compute_objF=="Default" or compute_gradF=="Default":
      return mysgd.my_sgd(w, h=h, c=c, maxiter=maxiter, ita=ita, Step_backtrack=Step_backtrack, stopMethod=stopMethod, itaOverIteration=itaOverIteration, tnot=tnot)
    else:
      return mysgd.my_sgd(w, compute_obj=compute_objF, compute_grad=compute_gradF, h=h, c=c, maxiter=maxiter, ita=ita, Step_backtrack=Step_backtrack, stopMethod=stopMethod, itaOverIteration=itaOverIteration, tnot=tnot)

  def fit(self, X, y, w, **kwargs):
    method = kwargs.get('method', 'gd')
    compute_obj = kwargs.get('compute_obj', 'Default')
    compute_grad = kwargs.get('compute_grad', 'Default')
    Step_backtrack = kwargs.get('Step_backtrack', False)
    stopMethod = kwargs.get('stopMethod', None) #user can specify the function
    h = kwargs.get('h', 0.3)
    c = kwargs.get('c', 1)
    maxiter = kwargs.get('maxiter', 100)
    ita = kwargs.get('ita', 0.11)
    itaOverIteration = kwargs.get('itaOverIteration', False)
    tnot = kwargs.get('tnot', 1)

    if isinstance(method, str) and method == 'gd':
      print "Running gradient descent"
      return self.gd(X, y, w, compute_obj, compute_grad, h=h, c=c, maxiter=maxiter, ita=ita, Step_backtrack=Step_backtrack, stopMethod=stopMethod)
    elif isinstance(method, str) and method == 'sgd':
      print "Running stochastic gradient descent"
      return self.sgd(X, y, w, compute_obj, compute_grad, h=h, c=c, maxiter=maxiter, 
        ita=ita, Step_backtrack=Step_backtrack, stopMethod=stopMethod, itaOverIteration=itaOverIteration, tnot=tnot)


# X, y = pt.dataset_fixed_cov(500, 10, 3) #n, dim, overlapped dist
# print "X.shape", X.shape
# min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
# X = min_max_scaler.fit_transform(X)
# rng = np.random.RandomState(19850920)
# permutation = rng.permutation(len(X))
# X, y = X[permutation], y[permutation]
# train_X, test_X, train_y, test_y = train_test_split(X, y, train_size=0.1, random_state=2010)
# # pt.plotPCA(X, y)

# n, dim = train_X.shape
# mysvm = svm()
# mygd = gd.gradientDescent(train_X, train_y)
# w = np.zeros(dim)
# averagedwIter, allw, w, iterCount = mysvm.fit(train_X, train_y, w, method="sgd", compute_obj="Default", compute_grad="Default", maxiter=3, ita=0.11, stopMethod="optimize")
# print w, iterCount
# print "averagedwIter[:5]\n", averagedwIter[:5]
# print "accuracy averagedwIter over iteration:\n", gd.getAccuracyOverIteration(averagedwIter, train_X, train_y)
# print "objective of averagedwIter over iteration:\n", map(mygd.compute_obj, averagedwIter)
# print "objective of allw over iteration:\n", map(mygd.compute_obj, allw)
