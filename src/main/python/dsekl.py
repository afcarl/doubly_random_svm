import os
import sys
import tempfile
import numpy as np
import scipy as sp
from numpy import load

import datetime
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.externals.joblib import Parallel, delayed, dump, load

def svm_gradient(X,y,w,n_pred_samples,n_expand_samples,C=.0001,sigma=1.,seed=1):
    if(seed < 0 or seed > 4294967295):
        # print "caution! seed is:", seed, "setting to 1"
        seed = sp.random.randint(0, 4294967295)
    sp.random.seed(seed)
    # sample Kernel
    rnpred = sp.random.randint(low=0,high=len(y),size=n_pred_samples)
    rnexpand = sp.random.randint(low=0,high=len(y),size=n_expand_samples)
    K = GaussKernMini(X[rnpred,:].T,X[rnexpand,:].T,sigma)
    # compute predictions
    yhat = K.dot(w[rnexpand])
    # compute whether or not prediction is in margin
    inmargin = (yhat * y[rnpred]) <= 1
    # compute gradient
    G = C * w[rnexpand] - (y[rnpred] * inmargin).dot(K)
    return G,rnexpand


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
    if (seed < 0 or seed > 4294967295):
        # print "caution! seed is:", seed, "setting to 1"
        seed = sp.randomrandint(0,4294967295)
    sp.random.seed(seed)
    rnexpand = sp.random.randint(low=0,high=Xtrain.shape[0],size=n_expand_samples)
    K = GaussKernMini(Xtest.T,Xtrain[rnexpand,:].T,sigma)
    # compute predictions
    return K.dot(w[rnexpand])


def svm_predict_raw_back(Xtrain,Xtest,w,n_expand_samples,sigma=1.,seed=0):
    # sample Kernel
    if (seed < 0 or seed > 4294967295):
        print "caution! seed is:", seed, "setting to 1"
        seed = 1
    sp.random.seed(seed)
    rnexpand = sp.random.randint(low=0,high=Xtrain.shape[0],size=n_expand_samples)
    K = GaussKernMini(Xtest.T,Xtrain[rnexpand,:].T,sigma)
    # compute predictions
    return K.dot(w[rnexpand])

def svm_predict_part(Xtrain,Xtest,w,rnexpand,sigma=1.,seed=0):
    # sample Kernel
    K = GaussKernMini(Xtest.T,Xtrain[rnexpand,:].T,sigma)
    # compute predictions
    return K.dot(w[rnexpand])

def svm_predict_all(Xtrain,Xtest,w,gamma,sigma=1):
    K = GaussKernMini(Xtest.T,Xtrain.T,sigma)
    # compute predictions
    return K.dot(w)


class DSEKL(BaseEstimator, ClassifierMixin):
    """
    Doubly Stochastic Empirical Kernel Learning (for now only with SVM and RBF kernel)
    """
    def __init__(self,n_expand_samples=100,n_pred_samples=100,n_its=100,eta=1.,C=.001,gamma=1.,workers=1,damp=True,validation=False,verbose=False):
        self.n_expand_samples=n_expand_samples
        self.n_pred_samples=n_pred_samples
        self.n_its = n_its
        self.eta = eta
        self.C = C
        self.gamma = gamma
        self.workers = workers
        self.verbose = verbose
        self.damp = damp
        self.validation = validation
        pass

    def fit(self, X, y):
        idx = np.random.permutation(len(y))

        if self.validation:
            traintestsplit = len(y)*.01
            validx = idx[-traintestsplit:]
            trainidx = idx[:-traintestsplit]
            Xval = X[validx,:].copy()
            Yval = y[validx].copy()
            X = X[trainidx,:]
            y = y[trainidx]

        self.X = X
        self.y = y

        if self.verbose:
            if self.validation:
                print "Training DSEKL on %d samples and validating on %d"%(len(idx),traintestsplit)
            else:
                print "Training DSEKL on %d samples" % (len(idx))
            print "using %i workers"%(self.workers)

        self.classes_ = sp.unique(y)
        assert(all(self.classes_==[-1.,1.]))


        # w[:] = sp.float128(sp.randn(len(y)))
        w = sp.float128(sp.randn(len(y)))
        G = sp.ones(len(y))
        if self.validation:
            self.valErrors = []

        self.trainErrors = []
        self.w = w.copy()
        oldw = w.copy()
        it = 0
        # for it in range(self.n_its/self.workers):
        delta_w = 5
        while(it < int(self.n_its / self.workers)):
            it += 1

            if self.verbose:
                print "iteration %i of %0.2f" % (it, int(self.n_its / self.workers))
                if it * self.workers % 1 == 0:
                    # # train_error = (sp.sign(svm_predict_all(self.X, X, w, self.gamma)) != self.y).mean()
                    # # val_error = (sp.sign(svm_predict_all(self.X, Xval, w, self.gamma)) != Yval).mean()
                    # train_error = (sp.sign(self.transform(self.X)) != self.y).mean()
                    if self.validation:
                        val_error = (sp.sign(self.predict(Xval)) != Yval).mean()
                        self.valErrors.append(val_error)
                        print "Validation-Error: %0.2f"%(val_error)
                    # self.trainErrors.append(train_error)
                    # print "%i iterations Train-Error: %0.2f Validation-Error: %0.2f, change w: %0.2f" % \
                    #       (it,
                    #        train_error,
                    #        val_error,
                    #        sp.linalg.norm(oldw - w))

                    print datetime.datetime.now()
                    print "%i iterations, change w: %0.2f" % \
                          (it,
                           sp.linalg.norm(oldw - w))

            oldw = w.copy()
            seeds = np.random.randint(0, high=4294967295, size=self.workers)
            gradients = Parallel(n_jobs=-1)(delayed(svm_gradient)(self.X, self.y, \
                                                                       w.copy(), self.n_pred_samples, self.n_expand_samples, C=self.C, sigma=self.gamma, seed=seeds[i]) for i in range(self.workers))

            tmpw = sp.zeros(len(y))
            for g in gradients:
                if self.damp:
                    G[g[1]] += g[0]**2
                tmpw[g[1]] += g[0]

            for i in tmpw.nonzero()[0]:
                if self.damp:
                    w[i] -= self.eta * tmpw[i] / sp.sqrt(G[i])
                else:
                    w[i] -= tmpw[i] #/ sp.sqrt(G[i])

            self.eta = self.eta * self.eta
            self.w = w.copy()

        self.w = w.copy()
        return self


    def predict(self, Xtest):
        number_of_redraws = self.workers

        yraw = Parallel(n_jobs=-1)(delayed(svm_predict_raw)(self.X, Xtest, \
                                                            self.w, self.n_expand_samples, self.gamma, i) for i in
                                   range(number_of_redraws))

        yhat = sp.sign(sp.vstack(yraw).mean(axis=0))
        return yhat



    def predict_all_subsample(self,Xtest):
        # do in several steps:
        n = int(self.w.shape[0]/10000.0)
        step = []

        print "evaluating on n:",n,"steps"
        l = int(self.w.shape[0]/float(n))
        for i in range(0,n):
            step.append(range(i * l,(i+1) * l,1000))

        yraw = Parallel(n_jobs=-1)(delayed(svm_predict_part)(self.X, Xtest, \
                                                                self.w, step[i], self.gamma) for i in
                                       range(n))
        yhat = sp.sign(sp.vstack(yraw).mean(axis=0))
        return yhat


    def predict_support(self, Xtest):
        # percentile_1 = np.percentile(self.w,10)
        # percentile_2 = np.percentile(self.w,90)
        # idx = np.append(np.where(self.w > percentile_2)[0], np.where(self.w < percentile_1))

        print "self.w.shape[0] = ",self.w.shape[0]

        idx = np.append(np.where(self.w > 0.1)[0], np.where(self.w < -0.1))

        print "idx.shape[0]", idx.shape[0]
        # print percentile_1,percentile_2
        # print "found %i support vectors percentile1: %0.2f percentile2: %0.2f" % (idx.shape[0],percentile_1,percentile_2)

        # cut off if too many
        if(idx.shape[0] > 400):
            idx = np.random.permutation(idx)[:400]
        print "found %i support vectors" % (idx.shape[0])
        K = GaussKernMini(Xtest.T, self.X[idx, :].T, self.gamma)
        # compute predictions
        return K.dot(self.w[idx])

    def predict_all(self, Xtest):
        K = GaussKernMini(Xtest.T, self.X.T, self.gamma)
        return K.dot(self.w)

    def transform(self, Xtest): return self.predict(Xtest)#self.predict_all_subsample(Xtest)#self.predict(Xtest)#svm_predict_all(self.X,Xtest,self.w,self.gamma)#