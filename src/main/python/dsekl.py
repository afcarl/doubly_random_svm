import os
import sys
import tempfile
import numpy as np
import scipy as sp
from numpy import load

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.externals.joblib import Parallel, delayed, dump, load

def svm_gradient(X,y,w,n_pred_samples,n_expand_samples,C=.0001,sigma=1.,seed=1):
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
    sp.random.seed(seed)
    rnexpand = sp.random.randint(low=0,high=Xtrain.shape[0],size=n_expand_samples)
    K = GaussKernMini(Xtest.T,Xtrain[rnexpand,:].T,sigma)
    # compute predictions
    return K.dot(w[rnexpand])


def svm_predict_raw_back(Xtrain,Xtest,w,n_expand_samples,sigma=1.,seed=0):
    # sample Kernel
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
    def __init__(self,n_expand_samples=100,n_pred_samples=100,n_its=100,eta=1.,C=.001,gamma=1.,workers=1):
        self.n_expand_samples=n_expand_samples
        self.n_pred_samples=n_pred_samples
        self.n_its = n_its
        self.eta = eta
        self.C = C
        self.gamma = gamma
        self.workers = workers
        pass

    def fit(self, X, y):
        idx = np.random.permutation(len(y))
        # traintestsplit = len(y)*.2
        # testidx = idx[-traintestsplit:]
        # trainidx = idx[:-traintestsplit]
        # Xtest = X[testidx,:].copy()
        # Ytest = y[testidx].copy()
        # X = X[trainidx,:]
        # y = y[trainidx]

        traintestsplit = len(y)*.001
        validx = idx[-traintestsplit:]
        trainidx = idx[:-traintestsplit]
        Xval = X[validx,:].copy()
        Yval = y[validx].copy()
        X = X[trainidx,:]
        y = y[trainidx]



        print "Training DSEKL on %d samples and validating on %d"%(len(idx),traintestsplit)
        print "using %i workers"%(self.workers)

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
        w = np.memmap(w_name, dtype=sp.float128, shape=(len(y)), mode='w+')

        w[:] = sp.float128(sp.randn(len(y)))
        G = sp.ones(len(y))
        self.valErrors = []
        self.trainErrors = []
        self.w = w.copy()
        for it in range(self.n_its/self.workers):

            oldw = w.copy()
            # print "iteration %i of %0.2f" % (it, int(self.n_its / self.workers))
            # if it * self.workers % 100 == 0:
            #     # train_error = (sp.sign(svm_predict_all(self.X, X, w, self.gamma)) != self.y).mean()
            #     # val_error = (sp.sign(svm_predict_all(self.X, Xval, w, self.gamma)) != Yval).mean()
            #     train_error = (sp.sign(self.transform(self.X)) != self.y).mean()
            #     val_error = (sp.sign(self.transform(Xval)) != Yval).mean()
            #     self.valErrors.append(val_error)
            #     self.trainErrors.append(train_error)
            #     print "%i iterations Train-Error: %0.2f Validation-Error: %0.2f, change w: %0.2f" % \
            #           (it,
            #            train_error,
            #            val_error,
            #            sp.linalg.norm(oldw - w))

            seeds = np.random.randint(0, high=sys.maxint, size=self.workers)
            gradients = Parallel(n_jobs=-1)(delayed(svm_gradient)(self.X, self.y, \
                                                                       w.copy(), self.n_pred_samples, self.n_expand_samples, C=self.C, sigma=self.gamma, seed=seeds[i]) for i in range(self.workers))

            tmpw = sp.zeros(len(y))
            for g in gradients:
                G[g[1]] += g[0]**2
                tmpw[g[1]] += g[0]

            for i in tmpw.nonzero()[0]:
                w[i] -= tmpw[i] / sp.sqrt(G[i])

            self.w = w.copy()


                # print "%i Train-Error: %0.2f, change w: %0.2f" % \
                #             (it,
                #             val_error,
                #             sp.linalg.norm(oldw-w))



            # if it*self.workers % 1 == 0:
            # if it % 10 == 0:
            #     train_error = (sp.sign(svm_predict_all(self.X,X,w,self.gamma))!=self.y).mean()
            #     val_error = (sp.sign(svm_predict_all(self.X,Xtest,w,self.gamma))!=Ytest).mean()
            #     # train_error = (sp.sign(svm_predict_raw(self.X,X,w,self.n_expand_samples,sigma=self.gamma))!=self.y).mean()
            #     # val_error = (sp.sign(svm_predict_raw(self.X,Xtest,w,self.n_expand_samples,sigma=self.gamma))!=Ytest).mean()
            #     self.valErrors.append(val_error)
            #     self.trainErrors.append(train_error)
            #     print "%i Train-Error: %0.2f Test-Error: %0.2f, change w: %0.2f"%\
            #           (it,
            #            train_error,
            #            val_error,
            #            sp.linalg.norm(oldw-w))

        self.w = w.copy()
        return self


    def predict(self, Xtest):
        yraw = Parallel(n_jobs=-1)(delayed(svm_predict_raw)(self.X, Xtest, \
                                                            self.w, self.n_expand_samples, self.gamma, i) for i in
                                   range(1))

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
        idx = np.append(np.where(self.w > 1)[0], np.where(self.w < -1))
        # print percentile_1,percentile_2
        # print "found %i support vectors percentile1: %0.2f percentile2: %0.2f" % (idx.shape[0],percentile_1,percentile_2)
        idx = idx[::500]
        # cut off if too many
        if(idx.shape[0] > 400):
            idx = idx[:400]
        print "found %i support vectors" % (idx.shape[0])
        K = GaussKernMini(Xtest.T, self.X[idx, :].T, self.gamma)
        # compute predictions
        return K.dot(self.w[idx])

    def transform(self, Xtest): return self.predict_support(Xtest)#self.predict_all_subsample(Xtest)#self.predict(Xtest)#svm_predict_all(self.X,Xtest,self.w,self.gamma)#