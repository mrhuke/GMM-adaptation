#!/usr/bin/python

from sklearn.mixture import GMM
import pylab as pl
from scipy import linalg
import numpy as np
import matplotlib as mpl
import itertools
import csv

def gmm_mllr_diag_cov(X, gmm, niter=10):
  """ 
     GMM adaptation (only means) using MLLR for GMM with diagonal covariance matrix.

     Usage: gmm_mllr_diag_cov(X, gmm, niter)
     
     References: Leggetter & Woodland'1995
  """

  # remove illed gaussians
  logprob,pcompx = gmm.eval(X)
  psum = np.sum(pcompx, axis=0)
  ill_g = (psum == 0);
  if any(ill_g):
    valid = psum > 0
    gmm.means_ = gmm.means_[valid,:]
    gmm.weights_ = gmm.weights_[valid]
    gmm.weights_ = gmm.weights_/sum(gmm.weights_)
    gmm.covars_ = gmm.covars_[valid]
    logprob,pcompx = gmm.eval(X)

  # calculate G and Z 
  C = len(gmm.weights_)
  T,dim = X.shape
  W = np.empty([dim,dim+1])
  G = np.zeros([dim,dim+1,dim+1])
  # 1. first calculate D[0,...,C) and Z
  D = np.zeros([C,dim+1,dim+1])
  V = np.zeros([C,dim,dim])
  Z = np.zeros([dim,dim+1])
  for c in range(0,C):
    mu = gmm.means_[c]
    sigma = np.diag(gmm.covars_[c])
    sigma_inv = np.linalg.inv(sigma)
    p = pcompx[:,c]

    xi = np.empty_like(mu)
    xi[:] = mu
    xi = np.insert(xi,0,1)
    xi = np.reshape(xi, [len(xi),1])
    D[c] = xi.dot(xi.T)
    V[c] = np.sum(p)*sigma_inv
    for t in range(0,T):
      xt = np.reshape(X[t],[len(X[t]),1])
      Z += p[t]*sigma_inv.dot(xt).dot(xi.T)
    
  # 2. now calculate G
  for i in range(0,dim): 
    for c in range(0,C): # tie all Gaussians
      G[i] += V[c,i,i]*D[c]
    try: 
      G_i_inv = np.linalg.inv(G[i])
    except:
      print 'G is nearly singular and pseudo-inverse is used.'
      G_i_inv = np.linalg.pinv(G[i])
    z_i = np.reshape(Z[i],[dim+1,1])
    W[i] = G_i_inv.dot(z_i)[:,0]
  
  # transform means
  for c in range(0,C):
    xi = np.insert(gmm.means_[c],0,1)
    xi = np.reshape(xi,[len(xi),1])
    gmm.means_[c] = W.dot(xi)[:,0]
      
  # remove non positive definite matrices
  logprob,pcompx = gmm.eval(X)
  psum = np.sum(pcompx, axis=0)
  ill = (psum == 0);
  if np.any(ill):
    valid = (ill == 0)
    gmm.means_ = gmm.means_[valid]
    gmm.weights_ = gmm.weights_[valid]
    gmm.weights_ = gmm.weights_/sum(gmm.weights_)
    gmm.covars_ = gmm.covars_[valid]
    K = gmm.means_.shape[0]  
 
  return gmm

def plotgmm(gmm,X):
  color_iter = itertools.cycle(['r', 'g', 'b', 'c', 'm'])

  for i, (clf, title) in enumerate([(gmm, 'GMM')]):
    splot = pl.subplot(1, 1, 1 + i)
    Y_ = clf.predict(X)
    for i, (mean, covar, color) in enumerate(zip(
            clf.means_, clf._get_covars(), color_iter)):
        v, w = linalg.eigh(covar)
        u = w[0] / linalg.norm(w[0])
        # as the DP will not use every component it has access to
        # unless it needs it, we shouldn't plot the redundant
        # components.
        if not np.any(Y_ == i):
            continue
        pl.scatter(X[Y_ == i, 0], X[Y_ == i, 1], .8, color=color)

        # Plot an ellipse to show the Gaussian component
        angle = np.arctan(u[1] / u[0])
        angle = 180 * angle / np.pi  # convert to degrees
        ell = mpl.patches.Ellipse(mean, v[0], v[1], 180 + angle, color=color)
        #ell.set_clip_box(splot.bbox)
        ell.set_alpha(0.5)
        splot.add_artist(ell)

    pl.xlim(-10, 10)
    pl.ylim(-10, 10)
    pl.xticks(())
    pl.yticks(())
    pl.title(title)

  pl.show()

def main():
  n_samples = 1000

  # Generate random sample, two components
  np.random.seed(0)
  C = np.array([[0., -0.1], [1.7, .4]])
  X = np.r_[np.random.randn(n_samples,2)*2, np.random.randn(n_samples,2)*2+np.array([4,3])]
  Y = np.r_[np.random.randn(n_samples,2)*2+np.array([-4,3]), np.random.randn(n_samples,2)*2+np.array([5,-1])]
 
  # Fit an initial GMM
  gmm = GMM(n_components=3, covariance_type='diag')
  gmm.fit(X)
  plotgmm(gmm,X)
  plotgmm(gmm,Y)
  
  # MLLR adaptation
  for loop in np.arange(0,10):
    gmm = gmm_mllr_diag_cov(Y, gmm) 
    plotgmm(gmm,Y)

if __name__=='__main__':
  main()
