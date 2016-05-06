import numpy as np
import os
import scipy as sp

import datetime
import tempfile

from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.externals.joblib import Parallel, delayed, dump, load
from sklearn.utils import shuffle
from sklearn.utils.extmath import fast_dot

from sklearn.utils.validation import NonBLASDotWarning
np.warnings.simplefilter('always', NonBLASDotWarning)

home_dir_server = "/data/users/nsteenbergen/"
if not os.path.isdir(home_dir_server):
    home_dir_server = tempfile.mkdtemp()

def svm_gradient(X,y,w,n_pred_samples,n_expand_samples,C=.0001,sigma=1.,seed=1):
    if(seed < 0 or seed > 4294967295):
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

def svm_gradient_batch(X_pred,X_exp,y,X_pred_ids,X_exp_ids,w,C=.0001,sigma=1.):
    # sample Kernel
    rnpred = X_pred_ids#sp.random.randint(low=0,high=len(y),size=n_pred_samples)
    rnexpand = X_exp_ids#sp.random.randint(low=0,high=len(y),size=n_expand_samples)
    K = GaussKernMini(X_pred.T,X_exp.T,sigma)
    # compute predictions

    yhat = fast_dot(K,w[rnexpand])
    # compute whether or not prediction is in margin
    inmargin = (yhat * y[rnpred]) <= 1
    # compute gradient
    G = C * w[rnexpand] - fast_dot((y[rnpred] * inmargin), K)
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
    # K = sp.exp(-(G + H.T - 2.*fast_dot(X1.T,X2))/(2.*sigma**2))
    if sp.sparse.issparse(X1) | sp.sparse.issparse(X2): K = sp.array(K)
    return K

def GaussKernMini_fast(X1,X2,sigma):
    if sp.sparse.issparse(X1):
        G = sp.outer(X1.multiply(X1).sum(axis=0),sp.ones(X2.shape[1]))
    else:
        G = sp.outer((X1 * X1).sum(axis=0),sp.ones(X2.shape[1]))
    if sp.sparse.issparse(X2):
        H = sp.outer(X2.multiply(X2).sum(axis=0),sp.ones(X1.shape[1]))
    else:
        H = sp.outer((X2 * X2).sum(axis=0),sp.ones(X1.shape[1]))
    K = sp.exp(-(G + H.T - 2.*fast_dot(X1.T,X2))/(2.*sigma**2))
    # K = sp.exp(-(G + H.T - 2.*(X1.T.dot(X2)))/(2.*sigma**2))
    if sp.sparse.issparse(X1) | sp.sparse.issparse(X2): K = sp.array(K)
    return K

def svm_gradient_batch_fast(X_pred, X_exp, y, X_pred_ids, X_exp_ids, w, C=.0001, sigma=1.):
    # sample Kernel
    rnpred = X_pred_ids#sp.random.randint(low=0,high=len(y),size=n_pred_samples)
    rnexpand = X_exp_ids#sp.random.randint(low=0,high=len(y),size=n_expand_samples)
    #K = GaussKernMini_fast(X_pred.T,X_exp.T,sigma)
    X1 = X_pred.T
    X2 = X_exp.T
    if sp.sparse.issparse(X1):
        G = sp.outer(X1.multiply(X1).sum(axis=0), sp.ones(X2.shape[1]))
    else:
        G = sp.outer((X1 * X1).sum(axis=0), sp.ones(X2.shape[1]))
    if sp.sparse.issparse(X2):
        H = sp.outer(X2.multiply(X2).sum(axis=0), sp.ones(X1.shape[1]))
    else:
        H = sp.outer((X2 * X2).sum(axis=0), sp.ones(X1.shape[1]))
    K = sp.exp(-(G + H.T - 2. * fast_dot(X1.T, X2)) / (2. * sigma ** 2))
    # K = sp.exp(-(G + H.T - 2.*(X1.T.dot(X2)))/(2.*sigma**2))
    if sp.sparse.issparse(X1) | sp.sparse.issparse(X2): K = sp.array(K)

    # compute predictions
    yhat = fast_dot(K,w[rnexpand])
    # compute whether or not prediction is in margin
    inmargin = (yhat * y[rnpred]) <= 1
    # compute gradient
    G = C * w[rnexpand] - fast_dot((y[rnpred] * inmargin), K)
    return G,rnexpand



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


def svm_predict_raw_batches(Xtrain, Xtest, w, ids_expand, sigma=1.):
    K = GaussKernMini(Xtest.T,Xtrain[ids_expand,:].T, sigma)
    return K.dot(w[ids_expand])

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
        self.eta_start = eta
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

        w = sp.float128(sp.randn(len(y)))
        G = sp.ones(len(y))
        if self.validation:
            self.valErrors = []

        self.trainErrors = []
        self.w = w.copy()
        oldw = w.copy()
        it = 0



        data_name = os.path.join(home_dir_server, 'data')
        dump(X, data_name)
        self.X = load()
        self.X = load(data_name, mmap_mode='r')
        target_name = os.path.join(home_dir_server, 'target')
        dump(y, target_name)
        self.y = load(target_name, mmap_mode='r')
        w_name = os.path.join(home_dir_server, 'weights')
        w = np.memmap(w_name, dtype=sp.float128, shape=(len(y)), mode='w+')

        delta_w = 5
        while(it < int(self.n_its / self.workers) and delta_w > 1.):
            it += 1

            if self.verbose and it != 1:
                print "iteration %i of %0.2f" % (it, int(self.n_its / self.workers))
                if it * self.workers % 1 == 0:
                    if self.validation:
                        val_error = (sp.sign(self.predict(Xval)) != Yval).mean()
                        self.valErrors.append(val_error)
                        print "Validation-Error: %0.2f, discount: %0.10f"%(val_error,self.eta)

                    print datetime.datetime.now()
                    print "%i iterations, dicscount: %0.10f change w: %0.2f" % \
                          (it,
                           self.eta,
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
                    w[i] -= self.eta * tmpw[i] /float(len(gradients))#/ sp.sqrt(G[i])

            self.eta = self.eta * self.eta_start
            self.w = w.copy()
            delta_w = sp.linalg.norm(oldw - w)
        self.w = w.copy()
        return self


    '''
    method used now for prediction
    '''
    def predict(self, Xtest,number_of_redraws=1000):
        yraw = Parallel(n_jobs=8)(delayed(svm_predict_raw)(self.X, Xtest, \
                                                            self.w, self.n_expand_samples, self.gamma, i) for i in
                                   range(number_of_redraws))

        yhat = sp.sign(sp.vstack(yraw).mean(axis=0))
        return yhat

    '''
    try to evaluate on all points
    '''
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

    '''
    tries to smartly preselect support vectors
    '''
    def predict_support_hardlimits(self, Xtest):
        print "self.w.shape[0] = ",self.w.shape[0]

        idx = np.append(np.where(self.w > 0.1)[0], np.where(self.w < -0.1))

        print "idx.shape[0]", idx.shape[0]

        # cut off if too many
        if(idx.shape[0] > 400):
            idx = np.random.permutation(idx)[:400]
        print "found %i support vectors" % (idx.shape[0])
        K = GaussKernMini(Xtest.T, self.X[idx, :].T, self.gamma)
        # compute predictions
        return K.dot(self.w[idx])

    '''
    tries to smartly preselect support vectors
    '''
    def predict_support_percentiles(self, Xtest):
        percentile_1 = np.percentile(self.w,10)
        percentile_2 = np.percentile(self.w,90)
        idx = np.append(np.where(self.w > percentile_2)[0], np.where(self.w < percentile_1))

        print "self.w.shape[0] = ",self.w.shape[0]

        print "idx.shape[0]", idx.shape[0]

        print "found %i support vectors" % (idx.shape[0])
        K = GaussKernMini(Xtest.T, self.X[idx, :].T, self.gamma)
        # compute predictions
        return K.dot(self.w[idx])

    def predict_all(self, Xtest):
        K = GaussKernMini(Xtest.T, self.X.T, self.gamma)
        return K.dot(self.w)

    def transform(self, Xtest): return self.predict(Xtest)


class DSEKLBATCH(BaseEstimator, ClassifierMixin):
    """
    Doubly Stochastic Empirical Kernel Learning (for now only with SVM and RBF kernel)
    """
    def __init__(self,n_expand_samples=100,n_pred_samples=100,n_its=100,eta=1.,C=.001,gamma=1.,workers=1,damp=True,validation=False,verbose=False):
        self.n_expand_samples=n_expand_samples
        self.n_pred_samples=n_pred_samples
        self.n_its = n_its
        self.eta_start = eta
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
            traintestsplit = len(y)*0.002
            validx = idx[-traintestsplit:]
            trainidx = idx[:-traintestsplit]
            Xval = X[validx,:].copy()
            Yval = y[validx].copy()
            X = X[trainidx,:]
            y = y[trainidx]

        # divide in batches

        # number of batches for all data
        num_batch_pred = X.shape[0] / float(self.n_pred_samples)
        num_batch_exp = X.shape[0] / float(self.n_expand_samples)
        self.X_pred_ids = []
        self.y_pred_ids = []
        for i in range(0,int(num_batch_pred)):
            self.X_pred_ids.append(range((i) * self.n_pred_samples, (i+1) * self.n_pred_samples))
            self.y_pred_ids.append(range((i) * self.n_pred_samples, (i+1) * self.n_pred_samples))
        if (num_batch_pred - int(num_batch_pred)) > 0:
            self.X_pred_ids.append(range(int(num_batch_pred) * self.n_pred_samples, X.shape[0]))
            self.y_pred_ids.append(range(int(num_batch_pred) * self.n_pred_samples, y.shape[0]))

        self.X_exp_ids = []
        self.y_exp_ids = []
        for i in range(0,int(num_batch_exp)):
            self.X_exp_ids.append(range((i) * self.n_expand_samples, (i+1) * self.n_expand_samples))
            self.y_exp_ids.append(range((i) * self.n_expand_samples, (i+1) * self.n_expand_samples))
        if (num_batch_exp - int(num_batch_exp)) > 0:
            self.X_exp_ids.append(range(int(num_batch_exp) * self.n_pred_samples, X.shape[0]))
            self.y_exp_ids.append(range(int(num_batch_exp) * self.n_pred_samples, y.shape[0]))
        self.X = X
        self.y = y

        if self.verbose:
            if self.validation:
                print "Training DSEKL on %d samples and validating on %d"%(len(idx),traintestsplit)
            else:
                print "Training DSEKL on %d samples" % (len(idx))

            print "\nhyperparameters:\nn_expand_samples: ",self.n_expand_samples,"\nn_pred_samples: ",self.n_pred_samples,"\neta: ",self.eta_start,"\nC: ",self.C,"\ngamma: ",self.gamma,"\nworker: ",self.workers,"\ndamp: ",self.damp,"\nvalidation: ",self.validation,"\n"

        self.classes_ = sp.unique(y)
        assert(all(self.classes_==[-1.,1.]))


        w = sp.float128(sp.randn(len(y)))
        G = sp.ones(len(y))
        if self.validation:
            self.valErrors = []

        self.trainErrors = []
        self.w = w.copy()
        oldw = w.copy()
        it = 0
        delta_w = 50
        while ( it < self.n_its and delta_w > 1.):
            it += 1

            if self.verbose and it != 1:
                print "iteration %i of %0.2f" % (it, self.n_its)
                if it * self.workers % 1 == 0:
                    if self.validation:
                        val_error = (sp.sign(self.predict(Xval)) != Yval).mean()
                        self.valErrors.append(val_error)
                        print "Validation-Error: %0.2f, discount: %0.10f"%(val_error,self.eta)


                    print datetime.datetime.now()
                    print "%i iterations, dicscount: %0.10f change w: %0.2f" % \
                          (it,
                           self.eta,
                           sp.linalg.norm(oldw - w))


            oldw = w.copy()


            for i in range(0,len(self.X_pred_ids)):
                self.X_pred_ids[i] = shuffle(self.X_pred_ids[i])
            self.X_pred_ids = shuffle(self.X_pred_ids)

            # print "X_pred_ids:\n",self.X_pred_ids
            for i in range(0,len(self.X_pred_ids)):
                X_pred_id = self.X_pred_ids[i]
                if self.verbose:
                    print "training on batch:",i, " of ",len(self.X_pred_ids), datetime.datetime.now()

                gradients = Parallel(n_jobs=self.workers,max_nbytes=None,verbose=50) (delayed(svm_gradient_batch_fast)(self.X[X_pred_id,:], self.X[X_exp_id],self.y,X_pred_id, X_exp_id,w.copy(), C=self.C, sigma=self.gamma) for X_exp_id in self.X_exp_ids)

                tmpw = sp.zeros(len(y))
                for g in gradients:
                    if self.damp:
                        G[g[1]] += g[0]**2
                    tmpw[g[1]] += g[0]

                for i in tmpw.nonzero()[0]:
                    if self.damp:
                        w[i] -= self.eta * tmpw[i] / sp.sqrt(G[i])
                    else:
                        w[i] -= self.eta * tmpw[i] /float(len(gradients))#/ sp.sqrt(G[i])

            self.eta = 1./float(it)#self.eta * self.eta_start
            self.w = w
            delta_w = sp.linalg.norm(oldw - w)
        self.w = w.copy()
        return self


    '''
    method used now for prediction
    '''
    def predict_old(self, Xtest,number_of_redraws=1000):
        # number_of_redraws = self.workers
        yraw = Parallel(n_jobs=8)(delayed(svm_predict_raw)(self.X, Xtest, \
                                                            self.w, self.n_expand_samples, self.gamma, i) for i in
                                   range(number_of_redraws))

        yhat = sp.sign(sp.vstack(yraw).mean(axis=0))
        return yhat

    '''
    predict in batches
    '''
    def predict(self, Xtest):

        print "starting predict with:",Xtest.shape[0]," samples : ",datetime.datetime.now()
        num_batches = Xtest.shape[0] / float(self.n_pred_samples)

        test_ids = []

        for i in range(0, int(num_batches)):
            test_ids.append(range((i) * self.n_pred_samples, (i + 1) * self.n_pred_samples))

        if (num_batches - int(num_batches)) > 0:
            test_ids.append(range(int(num_batches) * self.n_pred_samples, Xtest.shape[0]))

        # values = [(test_id, exp_ids) for test_id in test_ids for exp_ids in self.X_exp_ids]
        yhattotal = []
        for i in range(0,len(test_ids)):
            # print "computing result with batches:",i," of:",len(test_ids)
            yraw = Parallel(n_jobs=self.workers)(delayed(svm_predict_raw_batches)(self.X, Xtest[test_ids[i]], \
                                                           self.w, v, self.gamma) for v in self.X_exp_ids)
            yhat = sp.sign(sp.vstack(yraw).mean(axis=0))
            yhattotal.append(yhat)

        yhattotal = [item for sublist in yhattotal for item in sublist]
        print "stopping predict:", datetime.datetime.now()
        return yhattotal

    '''
    try to evaluate on all points
    '''
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

    '''
    tries to smartly preselect support vectors
    '''
    def predict_support_hardlimits(self, Xtest):
        print "self.w.shape[0] = ",self.w.shape[0]

        idx = np.append(np.where(self.w > 0.1)[0], np.where(self.w < -0.1))

        print "idx.shape[0]", idx.shape[0]

        # cut off if too many
        if(idx.shape[0] > 400):
            idx = np.random.permutation(idx)[:400]
        print "found %i support vectors" % (idx.shape[0])
        K = GaussKernMini(Xtest.T, self.X[idx, :].T, self.gamma)
        # compute predictions
        return K.dot(self.w[idx])

    '''
    tries to smartly preselect support vectors
    '''
    def predict_support_percentiles(self, Xtest):
        percentile_1 = np.percentile(self.w,10)
        percentile_2 = np.percentile(self.w,90)
        idx = np.append(np.where(self.w > percentile_2)[0], np.where(self.w < percentile_1))

        print "self.w.shape[0] = ",self.w.shape[0]

        print "idx.shape[0]", idx.shape[0]
        print "found %i support vectors" % (idx.shape[0])
        K = GaussKernMini(Xtest.T, self.X[idx, :].T, self.gamma)
        # compute predictions
        return K.dot(self.w[idx])

    def predict_all(self, Xtest):
        K = GaussKernMini(Xtest.T, self.X.T, self.gamma)
        return K.dot(self.w)

    def transform(self, Xtest): return self.predict(Xtest)
