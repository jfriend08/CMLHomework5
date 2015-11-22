import numpy as np
import random, sys
import points as pt
import librosa
import cPickle as pickle
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from  scipy.spatial.distance import euclidean
from librosa.util import normalize
from pylab import plt

def selectGenre(X, y, selectGenre):
  selectGenre = 1
  selecty = y[y==selectGenre]
  selectX = X[y==selectGenre]
  remainy = y[y!=selectGenre]
  remainX = X[y!=selectGenre]

  rng = np.random.RandomState(0)
  permutation = rng.permutation(len(remainX))
  X_new = np.r_[selectX,remainX[permutation][:len(selectX)]]
  y_new = np.r_[selecty,-1*np.ones(len(selectX))]

  return X_new, y_new

def getData(path, **kwargs):
  selectGenre = kwargs.get('selectGenre', None)
  samples = pickle.load( open( path, "rb" ) )
  X = []
  y = []
  for genere_idx in xrange(len(samples.keys())):
    genere = samples.keys()[genere_idx]
    songs = samples[genere]
    for song_idx in xrange(len(songs)):
      song = songs[song_idx]
      song = [clip[:66000] for clip in song] #some clip has different num of signal
      X.append(song)
      y.append(genere_idx)
  print "X size/number of songs:", len(X)
  print "Number of clips per song:", len(X[0])
  print "y size:", len(y)
  return np.array(X), np.array(y)

def featureExtraction(X, transpose=True):
  def MFCC(signal, sr=22050):
    return librosa.feature.mfcc(y=np.array(signal), sr=sr, n_mfcc=12)

  X = np.array([map(MFCC, song) for song in X])
  # X = np.array([[MFCC(clip) for clip in song] for song in X])
  print "After MFCC X.shape", X.shape
  X_train_flattened = [val for sublist in X for val in sublist]
  print "X_train_flattened.shape", np.array(X_train_flattened).shape
  # librosa.display.specshow(X_train_flattened[0], x_axis='time')
  # plt.colorbar()
  # plt.title('MFCC X_train_flattened[0]')
  # plt.tight_layout()
  # plt.show()

  if transpose:
    X_train_flattened = np.array(map(np.transpose, X_train_flattened))
    print "After transpose X_train_flattened.shape", X_train_flattened.shape

  X_train_flattened_norm = normalize(X_train_flattened, norm=2)
  X_train_flattened_norm_final = np.array([mfcc for clip in X_train_flattened_norm for mfcc in clip])
  return X_train_flattened_norm_final

def my_vlad(centers, X, y, transpose=True):
  def MFCC(signal, sr=22050):
    return librosa.feature.mfcc(y=np.array(signal), sr=sr, n_mfcc=12)

  def getClosestDist(centers, x, returnIdx=False):
    min_dis = sys.float_info.max
    shortestCenter_idx = -1
    numC, dim = centers.shape
    for eachC_idx in xrange(numC):
      eachC = centers[eachC_idx]
      dist = euclidean(eachC, x)
      if dist < min_dis:
        min_dis = dist
        shortestCenter_idx = eachC_idx
    return (min_dis if not returnIdx else shortestCenter_idx)

  def sub(centers, clip):
    cidx = getClosestDist(centers, clip, True)
    return clip-centers[cidx]

  print "vlad before X.shape", X.shape
  X = np.array([ map(MFCC, song) for song in X ])
  print "vlad after X.shape", X.shape
  if transpose:
    Xt = np.array([ [ np.transpose(clip) for clip in song] for song in X])
    # Xt = np.array(map(np.transpose, X))
    print "After transpose Xt.shape", Xt.shape
    # print Xt
  X_subtractC = np.array([ np.sum(map(lambda eachtime: sub(centers, eachtime), clip), axis=0) for song in Xt for clip in song])
  # X_subtractC = np.array([ np.sum(np.sum(map(lambda eachtime: sub(centers, eachtime), clip), axis=0) for clip in song ) for song in Xt])
  # X_subtractC = np.array([ [np.sum(map(lambda eachtime: sub(centers, eachtime), clip), axis=0) for clip in song ] for song in Xt])
  # X_subtractC = np.array([np.sum(map(lambda clip: sub(centers, clip), song), axis=0) for song in Xt])
  y_flatten = []
  for genre in y:
    tmp = [genre for i in xrange(len(X_subtractC)/len(y))]
    y_flatten.extend(tmp)

  return X_subtractC, np.array(y_flatten)



# centers = np.array(
#   [[ 0.12455026,  0.12589914, -0.12758974,  0.138313,   -0.10902273,  0.12275701,-0.15445272,  0.04824689, -0.09092256,  0.22765484,  0.00157361,  0.09746945],
#    [ 0.13949584,  0.12110779,  0.06468275,  0.10524045,  0.09770974,  0.15314491,0.06903171,  0.15934369, -0.01715376,  0.08812701,  0.02516624,  0.06107955],
#    [ 0.12602307,  0.13282984, -0.11469537,  0.14521176, -0.12193538,  0.08603053,-0.20165656,  0.06584635, -0.14003101,  0.10053826, -0.11766645,  0.09690807],
#    [ 0.13041553,  0.12676358, -0.06865276,  0.13890042, -0.091898,    0.12733449,-0.09334595,  0.18043392, -0.11214386,  0.11356557, -0.15927186,  0.06240254],
#    [ 0.14529536,  0.09225925,  0.00862161,  0.05767342,  0.01234768,  0.05260295,  -0.06143607,  0.03186934, -0.01837779, -0.05159241, -0.09900298, -0.10201332],
#    [ 0.12025221,  0.12779263, -0.14943618,  0.11613325, -0.1018837,   0.1238181, -0.07858684, -0.02594327, -0.13208168,  0.06386554,  0.05935377,  0.20633186],
#    [ 0.12135699,  0.13617754, -0.14722568,  0.07956322, -0.16947128,  0.12307407, -0.04534169,  0.07184431, -0.17814921, -0.05708589, -0.16669711,  0.07424351],
#    [ 0.115089,    0.13814819, -0.14351122,  0.14969473, -0.06354044,  0.12761665, -0.00318303,  0.00512991, -0.03940165,  0.05467843, -0.05587058, -0.03259541]
#    ])
# print "centers.shape", centers.shape, centers


# X, y = getData("../homework2/data/data_small5.in", )
# X_new, y_new = selectGenre(X, y, 0)

# print "X_new.shape", X_new.shape, "y_new.shape", y_new.shape
# print y_new

# rng = np.random.RandomState(1985)
# permutation = rng.permutation(len(X_new))
# X_new, y_new = X_new[permutation], y_new[permutation]
# train_X, test_X, train_y, test_y = train_test_split(X_new, y_new, train_size=0.75, random_state=2010)

# trainX_centerSubstract, trainy_centerSubstract = my_vlad(centers, train_X, train_y)
# testX_centerSubstract, testy_centerSubstract = my_vlad(centers, test_X, test_y)

# from sklearn.grid_search import GridSearchCV
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import MultinomialNB

# # # clf = MultinomialNB().fit(trainX_centerSubstract, train_y)
# # clf = KNeighborsClassifier().fit(trainX_centerSubstract, train_y)
# # predicted = clf.predict(testX_centerSubstract)
# # print "naive bayes", np.mean(predicted == test_y)

# param_grid = {'n_neighbors': np.arange(3, 10),'weights':('uniform', 'distance'), 'algorithm':('auto','ball_tree', 'kd_tree', 'brute') }
# np.set_printoptions(suppress=True)
# grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, verbose=3)

# print "trainX_centerSubstract.shape", trainX_centerSubstract.shape, "trainy_centerSubstract.shape", trainy_centerSubstract.shape
# print "testX_centerSubstract.shape", testX_centerSubstract.shape, "testy_centerSubstract.shape", testy_centerSubstract.shape
# print "train_y.shape", train_y.shape
# grid_search.fit(trainX_centerSubstract, trainy_centerSubstract)
# print "----------------", "Done grid search", "----------------"
# predict = grid_search.predict(testX_centerSubstract)
# print grid_search.score(testX_centerSubstract, testy_centerSubstract)
# print grid_search.best_params_






