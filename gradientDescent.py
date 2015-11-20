import numpy as np
import random, sys
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
    self.c = 1

  def computeYT(self, w):
    wt = map(lambda x: np.dot(w, x), self.X)
    return np.dot(self.y, wt)

  def compute_obj(self, w):
    yt = self.computeYT(w)
    n = float(len(self.y))
    if yt > 1+self.h:
      return np.dot(w, w)
    elif yt < 1-self.h:
      return np.dot(w, w) + (self.c/n)*(1-yt)
    else:
      return np.dot(w, w) + (self.c/n)*(1+self.h-yt)**2/4*self.h


x = np.array([np.ones(5), np.ones(5), np.ones(5), np.ones(5), np.ones(5), np.ones(5)])
y = np.array([-1, -1, 1, 1, -1, -1])

gd = gradientDescent(y, x)
w = np.array([1, 1, 0, 1, 0])
gd.compute_obj(w)





