#import pylab as pl
import scipy as sp
from scipy.stats.mstats import zscore
from numpy.random import multivariate_normal as mvn
import sklearn
import pdb
import sys
import os
import urllib
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import fetch_mldata,make_gaussian_quantiles
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from multiprocessing import Pool
import tempfile
from sklearn.externals.joblib.pool import has_shareable_memory
from sklearn.externals.joblib import Parallel, delayed, dump, load
import numpy as np
import shutil

custom_data_home = "/home/biessman/dskl/"

def GaussKernMini(X1,X2,sigma):
    if sp.sparse.issparse(X1):
        G = sp.outer(X1.multiply(X1).sum(axis=0),sp.ones(X2.shape[1]))
    else:
        G = sp.outer((X1 * X1).sum(axis=0),sp.ones(X2.shape[1]))
    if sp.sparse.issparse(X2):
        H = sp.outer(X2.multiply(X2).sum(axis=0),sp.ones(X1.shape[1]))
    else:
        H = sp.outer((X2 * X2).sum(axis=0),sp.ones(X1.shape[1]))
    K = sp.exp(-(G + H.T - 2.*(X1.T.dot(X2)))/(2.*sigma**2))
    if sp.sparse.issparse(X1) | sp.sparse.issparse(X2): K = sp.array(K)
    return K

def svm_predict_raw(Xtrain,Xtest,w,n_expand_samples,sigma=1.,seed=0):
    # sample Kernel
    sp.random.seed(seed)
    rnexpand = sp.random.randint(low=0,high=Xtrain.shape[0],size=n_expand_samples)
    K = GaussKernMini(Xtest.T,Xtrain[rnexpand,:].T,sigma)
    # compute predictions
    return K.dot(w[rnexpand]) 
    

def svm_gradient(X,y,w,n_pred_samples,n_expand_samples,C=.0001,sigma=1.):
    # sample Kernel
    rnpred = sp.random.randint(low=0,high=len(y),size=n_pred_samples)
    rnexpand = sp.random.randint(low=0,high=len(y),size=n_expand_samples)
    K = GaussKernMini(X[rnpred,:].T,X[rnexpand,:].T,sigma)
    # compute predictions
    yhat = K.dot(w[rnexpand])

    # compute whether or not prediction is in margin
    inmargin = (yhat * y[rnpred]) <= 1
    # compute gradient 
    G = (C * w[rnexpand] - (y[rnpred] * inmargin).dot(K) / (n_pred_samples * n_expand_samples))
    return G,rnexpand


class DSEKL(BaseEstimator, ClassifierMixin):
    """
    Doubly Stochastic Empirical Kernel Learning (for now only with SVM and RBF kernel)
    """
    def __init__(self,n_expand_samples=100,n_pred_samples=100,n_its=100,eta=1.,C=.001,gamma=1.,workers=1):
        self.n_expand_samples=n_expand_samples
        self.n_pred_samples=n_pred_samples
        self.n_its = n_its
        self.eta = eta
        self.C = C#1./(n_its * 100)
        self.gamma = gamma
        self.workers = workers
        pass

    def fit(self, X, y):
        idx = np.random.permutation(len(y))
        traintestsplit = len(y)*.2
        testidx = idx[-traintestsplit:]
        trainidx = idx[:-traintestsplit]
        Xtest = X[testidx,:].copy()
        Ytest = y[testidx].copy()
        X = X[trainidx,:]
        y = y[trainidx]
        print "Training DSEKL on %d samples, testing on %d samples"%(len(trainidx), len(testidx))
        self.classes_ = sp.unique(y)
        assert(all(self.classes_==[-1.,1.]))        
        folder = tempfile.mkdtemp()
        data_name = os.path.join(folder, 'data')
        dump(X, data_name)
        self.X = load(data_name, mmap_mode='r')
        target_name = os.path.join(folder, 'target')
        dump(y, target_name)
        self.y = load(target_name, mmap_mode='r')
        w_name = os.path.join(folder, 'weights')
        self.w = np.memmap(w_name, dtype=sp.float128, shape=(len(y)), mode='w+')

        #self.w[:] = sp.float128(sp.randn(len(y))) / len(y)
        G = sp.ones(len(y))/len(y)
        for it in range(self.n_its/self.workers):
            oldw = self.w.copy()
            gradients = Parallel(n_jobs=1)(delayed(svm_gradient)(self.X, self.y,\
                self.w, self.n_pred_samples, self.n_expand_samples, self.C, self.gamma) for i in range(self.workers))
            tmpw = sp.zeros(len(y))
            for g in gradients:
                G[g[1]] += g[0]**2
                tmpw[g[1]] += g[0]
            for i in tmpw.nonzero()[0]:
                self.w[i] -= tmpw[i] / sp.sqrt(G[i])
            if it % 1 == 0:
                ridx = range(Xtest.shape[0])#sp.random.permutation(Xtest.shape[0])[:100]
                #print "it: %d | Test-Error: %0.2f, change w: %0.2f"%(it,(sp.sign(svm_predict_raw(self.X,Xtest[ridx],w,self.n_expand_samples,self.gamma,0))!=Ytest[ridx]).mean(),sp.linalg.norm(oldw-w))
                print "it: %d | Test-Error: %0.2f, change w: %0.2f"%(it,(self.predict(Xtest[ridx,:])!=Ytest[ridx]).mean(),sp.linalg.norm(oldw-self.w))
        shutil.rmtree(folder)

        return self


    def predict(self, Xtest):
        yraw = Parallel(n_jobs=-1)(delayed(svm_predict_raw)(self.X, Xtest,\
                 self.w, self.n_expand_samples, self.gamma, i) for i in range(self.workers))
        yhat = sp.sign(sp.vstack(yraw).mean(axis=0))
        return yhat



    def transform(self, Xtest): return self.predict(Xtest)

def get_svmlight_file(fn):
    from sklearn.datasets import load_svmlight_file
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"+fn
    fn = os.path.join(custom_data_home,"svmlightdata",fn)
    if not os.path.isfile(fn):
        print("Downloading %s to %s"%(url,fn))
        urllib.urlretrieve(url,fn)
    return load_svmlight_file(fn)

def run_all_realdata(dnames=['sonar','mushroom','skin_nonskin','covertype','diabetes','gisette']):
    [run_realdata(dname=d) for d in dnames]

def load_realdata(dname="mushrooms"):
    print "Loading %s"%dname
    if dname == 'covertype':
        dd = fetch_mldata('covtype.binary', data_home=custom_data_home)
        Xtotal = dd.data
        Ytotal = sp.sign(dd.target - 1.5)
    elif dname == 'mnist':
        dd = sklearn.datasets.load_digits(2)
        Xtotal = dd.data
        Ytotal = sp.sign(dd.target - .5) 
    elif dname == 'breast':
        Xtotal,Ytotal = get_svmlight_file("breast-cancer_scale")
        Ytotal = Ytotal - 3 
    elif dname == 'diabetes':
        Xtotal,Ytotal = get_svmlight_file("diabetes_scale")
    elif dname == 'news':
        Xtotal,Ytotal = get_svmlight_file("news20.binary.bz2")
    elif dname == 'gisette':
        Xtotal,Ytotal = get_svmlight_file("gisette_scale.bz2")
    elif dname == 'skin_nonskin':
        Xtotal,Ytotal = get_svmlight_file("skin_nonskin")
        Ytotal = sp.sign(Ytotal - 1.5)
    elif dname == 'sonar':
        Xtotal,Ytotal = get_svmlight_file("sonar_scale")
    elif dname == 'mushrooms':
        Xtotal,Ytotal = get_svmlight_file("mushrooms")
        Ytotal = sp.sign(Ytotal - 1.5)
    elif dname == 'madelon':
        Xtotal,Ytotal = get_svmlight_file("madelon")
    return Xtotal,Ytotal

def run_realdata_no_comparison(dname='sonar',N=1000):
    Xtrain,Ytrain = load_realdata(dname)
    idx = sp.random.permutation(Xtrain.shape[0])
    DS = DSEKL(n_pred_samples=100,n_expand_samples=1000,n_its=N,C=1e-8,gamma=900.,workers=100).fit(Xtrain[idx[:N]],Ytrain[:N])

def run_realdata(reps=2,dname='sonar',maxN=1000):
    Xtotal,Ytotal = load_realdata(dname)

    params_dksl = {
            'n_pred_samples': [100],
            'n_expand_samples': [500],
            'n_its':[1000],
            'eta':[1.],
            'C':[1e-6],#10.**sp.arange(-8,-6,1),
            'gamma':[10.]#10.**sp.arange(-1.,2.,1)
            }
    
    params_batch = {
            'C':10.**sp.arange(-8,4,2),
            'gamma':10.**sp.arange(-4.,4.,2)
            }
 
 
    N = sp.minimum(Xtotal.shape[0],maxN)
    
    Eemp,Ebatch = [],[]

    for irep in range(reps):
        idx = sp.random.randint(low=0,high=Xtotal.shape[0],size=N)
        Xtrain = Xtotal[idx[:N/2],:]
        Ytrain = Ytotal[idx[:N/2]]
        Xtest = Xtotal[idx[N/2:],:]
        Ytest = Ytotal[idx[N/2:]]        
        if not sp.sparse.issparse(Xtrain): 
            scaler = StandardScaler()
            scaler.fit(Xtrain)  # Don't cheat - fit only on training data
            Xtrain = scaler.transform(Xtrain)
            Xtest = scaler.transform(Xtest)

        print "Training empirical"
        clf = GridSearchCV(DSEKL(),params_dksl,n_jobs=1,verbose=1,cv=2).fit(Xtrain,Ytrain)
        Eemp.append(sp.mean(clf.best_estimator_.transform(Xtest)!=Ytest))
        clf_batch = GridSearchCV(svm.SVC(),params_batch,n_jobs=-2,verbose=1,cv=2).fit(Xtrain,Ytrain)
        Ebatch.append(sp.mean(clf_batch.best_estimator_.predict(Xtest)!=Ytest))
        print "Emp: %0.2f - Batch: %0.2f"%(Eemp[-1],Ebatch[-1])
        print clf.best_estimator_.get_params()
        print clf_batch.best_estimator_.get_params()
    print "***************************************************************"
    print "Data set [%s]: Emp_avg: %0.2f+-%0.2f - Ebatch_avg: %0.2f+-%0.2f"%(dname,sp.array(Eemp).mean(),sp.array(Eemp).std(),sp.array(Ebatch).mean(),sp.array(Ebatch).std())
    print "***************************************************************"

if __name__ == '__main__':
    dname = sys.argv[1]
    N = int(sys.argv[2])
    nWorkers = int(sys.argv[3])
    nExpand = int(sys.argv[4])
    nits = int(sys.argv[5])
    cexp = int(sys.argv[6])
    Xtrain,Ytrain = load_realdata(dname)
    idx = sp.random.permutation(Xtrain.shape[0])
    DS = DSEKL(n_pred_samples=nExpand,n_expand_samples=nExpand,n_its=nits,C=10.**cexp,gamma=900.,workers=nWorkers).fit(Xtrain[idx[:N],:],Ytrain[:N])
