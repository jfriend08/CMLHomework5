import numpy as np
import random, sys
from sklearn import preprocessing
import matplotlib.pyplot as plt

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

class gradientDescent(object):
  def __init__(self, y, X):
    self.y = y
    self.X = X
    self.h = 0.3
    self.c = 2

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
    y, wx = myinput
    unit = np.ones(self.X.shape[1])
    dirwx = np.apply_along_axis(lambda x: np.dot(unit, x), 1, self.X)
    if y*wx > 1+self.h:
      return 0
    elif y*wx < 1-self.h:
      return -y*dirwx
    else:
      return (2*(1+self.h-y*wx)/(4*self.h))*(-1)*y*dirwx

  def compute_obj(self, w):
    n = self.X.shape[0]
    wx = np.apply_along_axis(lambda x: np.dot(w, x), 1, self.X)
    return np.dot(w, w) + (self.c/n)*sum(np.apply_along_axis(self.lossF, 1, zip(self.y, wx)) )


  def compute_grad(self, w):
    wx = np.apply_along_axis(lambda x: np.dot(w, x), 1, self.X)
    return 2*w + sum(np.apply_along_axis(lambda x: self.lossF_dir(w, self.X, x), 1, zip(self.y, wx)) )

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

min_max_scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1), copy=True)
x = np.array([np.zeros(5), np.zeros(5), np.ones(5), np.ones(5), np.zeros(5), np.zeros(5)])
x = min_max_scaler.fit_transform(x)
y = np.array([-1, -1, 1, 1, -1, -1])

gd = gradientDescent(y, x)
w = np.array([1, 1, 0, 1, 0])
gd.compute_obj(w)
gd.compute_grad(w)

for i in xrange(5):
  w = np.random.randint(100, size=(1, 5)).flatten()
  print i, "th checking: error sum", gd.grad_checker(gd.compute_obj, gd.compute_grad, w)



  # def compute_obj(self, w):
  #   yt = self.computeYT(w)
  #   n = float(len(self.y))
  #   if yt > 1+self.h:
  #     return np.dot(w, w)
  #   elif yt < 1-self.h:
  #     return np.dot(w, w) + (self.c/n)*(1-yt)
  #   else:
  #     return np.dot(w, w) + (self.c/n)*(1+self.h-yt)**2/4*self.h

  # def compute_grad(self, w):
  #   yt = self.computeYT(w)
  #   n = float(len(self.y))
  #   if yt > 1+self.h:
  #     return 2*w
  #   elif yt < 1-self.h:
  #     return 2*w - (self.c/n)*(self.computeYderT())
  #   else:
  #     return 2*w + (self.c/n) * ((2*(1+self.h-yt)/(4*self.h)) * self.computeYderT() )