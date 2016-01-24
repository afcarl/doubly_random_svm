import pylab as pl
import scipy as sp
from scipy.stats.mstats import zscore
from numpy.random import multivariate_normal as mvn
import sklearn
import pdb
import os
import urllib
from scipy.spatial.distance import cdist
from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.datasets import fetch_mldata,make_gaussian_quantiles
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler

custom_data_home = "/Users/biessman/Data/"

def GaussianKernel(X1, X2, sigma):
    assert(X1.shape[0] == X2.shape[0])
    if sp.sparse.issparse(X1): X1 = sp.array(X1.todense())
    if sp.sparse.issparse(X2): X2 = sp.array(X2.todense())
    K = cdist(X1.T, X2.T, 'euclidean')
    K = sp.exp(-(K ** 2) / (2. * sigma ** 2))
    return K

class DSEKL(BaseEstimator, ClassifierMixin):
    """
    Doubly Stochastic Empirical Kernel Learning (for now only with SVM and RBF kernel)
    """
    def __init__(self,n_expand_samples=100,n_pred_samples=100,n_its=100,eta=1.,C=.001,gamma=1.):
        self.n_expand_samples=n_expand_samples
        self.n_pred_samples=n_pred_samples
        self.n_its = n_its
        self.eta = eta
        self.C = C
        self.gamma = gamma
        pass

    def fit(self, X, y):
        self.classes_ = sp.unique(y)
        assert(all(self.classes_==[-1.,1.]))
        self.X = X
        self.y = y
        self.w = sp.float128(sp.randn(len(y)))

        for it in range(1,self.n_its+1):
            self.take_gradient_step(it)

        return self

    def predict(self, Xtest):
        K,rnexpand = self.sample_kernel_test(Xtest)
        return sp.sign(K.dot(self.w[rnexpand]) + 1e-30) 

    def sample_kernel(self):
        # get some random indices for predictions (the minibatch on which to compute the gradient)
        if self.n_pred_samples<1: rnpred=range(len(self.y))
        else: rnpred = sp.random.randint(low=0,high=len(self.y),size=self.n_pred_samples)
        K,rnexpand = self.sample_kernel_test(self.X[rnpred,:])
        return K,rnpred,rnexpand
    
    def sample_kernel_test(self,Xtest):
        # compute predictions for all test data
        rnpred=range(Xtest.shape[0])
        # get some random indices for expanding the empirical kernel map
        if self.n_expand_samples<1:rnexpand=range(len(self.y))
        else: rnexpand = sp.random.randint(low=0,high=len(self.y),size=self.n_expand_samples)
        # check if data matrix is sparse
        if sp.sparse.issparse(self.X): 
            Xexpand = sp.array(self.X[rnexpand,:].todense()).T
        else: 
            Xexpand = self.X[rnexpand,:].T
        # check if test data matrix is sparse
        if sp.sparse.issparse(Xtest): 
            Xtest = sp.array(Xtest.todense()).T
        else: 
            Xtest = Xtest.T
            
        # compute kernel for random sample
        K = GaussianKernel(Xtest,Xexpand,self.gamma)
        return K,rnexpand

    def take_gradient_step(self,step_size):
        # take random sample from kernel matrix
        K,rnpred,rnexpand = self.sample_kernel()
        # compute prediction 
        yhat = K.dot(self.w[rnexpand])    
        # compute whether or not prediction is in margin
        inmargin = (yhat * self.y[rnpred]) <= 1
        # compute gradient for 
        G = self.C * self.w[rnexpand] - (self.y[rnpred] * inmargin).dot(K)
        self.w[rnexpand] -= step_size * G
        return G

    def transform(self, Xtest): return self.predict(Xtest)

def get_svmlight_file(fn):
    from sklearn.datasets import load_svmlight_file
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"+fn
    fn = os.path.join(custom_data_home,"svmlightdata",fn)
    if not os.path.isfile(fn):
        print("Downloading %s to %s"%(url,fn))
        urllib.urlretrieve(url,fn)
    return load_svmlight_file(fn)

def run_all_realdata(dnames=['gisette','news','covertype','diabetes']):
    [run_realdata(dname=d) for d in dnames]

def run_realdata(reps=10,dname="covertype"):
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
        Xtotal = Xtotal
        Ytotal = Ytotal - 3 
    elif dname == 'diabetes':
        Xtotal,Ytotal = get_svmlight_file("diabetes_scale")
        Xtotal = Xtotal
        Ytotal = Ytotal
    elif dname == 'news':
        Xtotal,Ytotal = get_svmlight_file("news20.binary.bz2")
        Xtotal = Xtotal
        Ytotal = Ytotal
    elif dname == 'gisette':
        Xtotal,Ytotal = get_svmlight_file("gisette_scale.bz2")
        Xtotal = Xtotal
        Ytotal = Ytotal

    params = {
            'n_pred_samples': [100],
            'n_expand_samples': [1000],
            'n_its':[100],
            'eta':[1.],
            'C':10.**sp.arange(-6,2,2),
            'gamma':10.**sp.arange(-4,4,2)
            }
 
    N = sp.minimum(Xtotal.shape[1],1000)
    
    Eemp,Ebatch = [],[]

    for irep in range(reps):
        idx = sp.random.randint(low=0,high=Xtotal.shape[1],size=N)
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
        clf = GridSearchCV(DSEKL(),params).fit(Xtrain,Ytrain)
        Eemp.append(sp.mean(clf.best_estimator_.transform(Xtest)!=Ytest))
        clf_batch = GridSearchCV(svm.SVC(),{'C':params['C'],'gamma':params['gamma']}).fit(Xtrain,Ytrain)
        Ebatch.append(sp.mean(clf_batch.best_estimator_.predict(Xtest)!=Ytest))
        print "Emp: %0.2f - Batch: %0.2f"%(Eemp[-1],Ebatch[-1])
        print clf.best_estimator_.get_params()
        print clf_batch.best_estimator_.get_params()

    print "Emp_avg: %0.2f+-%0.2f - Ebatch_avg: %0.2f+-%0.2f"%(sp.array(Eemp).mean(),sp.array(Eemp).std(),sp.array(Ebatch).mean(),sp.array(Ebatch).std())

def fit_svm_kernel(X,Y,its=30,eta=1.,C=.1,kernel=(GaussianKernel,(1.)),nPredSamples=10,nExpandSamples=10):
    D,N = X.shape[0],X.shape[1]
    X = sp.vstack((sp.ones((1,N)),X))
    W = sp.randn(len(Y))
    for it in range(its):
        print "Iteration %4d Accuracy %0.3f"%(it,sp.mean(Y==sp.sign(kernel[0](X,X,kernel[1]).dot(W))))
        rnpred = sp.random.randint(low=0,high=N,size=nsamples)
        rnexpand = sp.random.randint(low=0,high=N,size=nsamples)
        # compute gradient 
        G = compute_gradient(Y[rnpred],X[:,rnpred],X[:,rnexpand],W[rnexpand],kernel,C)
        # update 
        W[rnexpand] -= eta/(it+1.) * G
	
    return W
def run_comparison_pred(N=100,features=2,nPredSamples=[1,20,50],its=100,reps=100):
    noise = 0.2
    C = .1
    eta = 1.
    nExpand = 20
    pl.ion()
    pl.figure(figsize=(6,4.5))
    colors = "brymcwg"
    leg = []
    for idx,cond in enumerate(nPredSamples):
        Eemp, Erks, Ebatch, EempFix = sp.zeros((reps,its)), sp.zeros((reps,its)), sp.zeros((reps,its)), sp.zeros((reps,its))
        for irep in range(reps):
            X,Y = make_data_xor(N*2,noise=noise)
            #X,Y = make_gaussian_quantiles(n_classes=2,n_samples=N*2,n_features=features)
            #Y=sp.sign(Y-.5)
            #X = X.T
            # split into train and test
            Xtest = X[:,:len(Y)/2]
            Ytest = Y[:len(Y)/2]
            X = X[:,(len(Y)/2):]
            Y = Y[(len(Y)/2):]

            EempFix[irep,:] = fit_svm_dskl_emp(X[:,:nExpand],Y[:nExpand],Xtest,Ytest,nExpandSamples=0,its=its,nPredSamples=cond,eta=eta,C=C)
            Eemp[irep,:] = fit_svm_dskl_emp(X,Y,Xtest,Ytest,nExpandSamples=nExpand,its=its,nPredSamples=cond,eta=eta,C=C)
            Erks[irep,:] = fit_svm_dskl_rks(X,Y,Xtest,Ytest,nExpandSamples=nExpand,its=its,nPredSamples=cond,eta=eta,C=C)
            Ebatch[irep,:] = fit_svm_batch(X,Y,Xtest,Ytest,1.)

        pl.clf()
        pl.plot(Erks.mean(axis=0),'k-')
        pl.plot(Eemp.mean(axis=0),'k.')
        pl.plot(EempFix.mean(axis=0),'k--')
        leg.append("Rks")
        leg.append("Emp")
        leg.append("Emp_{Fix}")
    
        pl.plot(Ebatch.mean()*sp.ones(Eemp.shape[1]),'k:')
        leg.append("Batch")
        if idx==0:pl.legend(leg,loc=3)
        pl.title("nPred=%d"%cond)
        pl.xlabel("Iterations") 
        pl.ylabel("Error")
        pl.axis('tight')
        pl.ylim((0,.55))
        #pl.axis('tight')
        pl.savefig("rks_emp_comparison-expand-%d-pred-%d.pdf"%(nExpand,cond))


def run_comparison_expand(N=100,features=4,nExpandSamples=[1,20,50],its=100,reps=100):
    noise = 0.2
    C = .1
    eta = 1.
    nPred = 20
    pl.ion()
    pl.figure(figsize=(6,4.5))
    colors = "brymcwg"
    leg = []
    for idx,cond in enumerate(nExpandSamples):
        Eemp, Erks, Ebatch, EempFix = sp.zeros((reps,its)), sp.zeros((reps,its)), sp.zeros((reps,its)), sp.zeros((reps,its))
        for irep in range(reps):
            X,Y = make_data_xor(N*2,noise=noise)
            #X,Y = make_gaussian_quantiles(n_classes=2,n_samples=N*2,n_features=features)
            #Y=sp.sign(Y-.5)
            #X = X.T
            # split into train and test
            Xtest = X[:,:len(Y)/2]
            Ytest = Y[:len(Y)/2]
            X = X[:,(len(Y)/2):]
            Y = Y[(len(Y)/2):]

            EempFix[irep,:] = fit_svm_dskl_emp(X[:,:cond],Y[:cond],Xtest,Ytest,nExpandSamples=0,its=its,nPredSamples=nPred,eta=eta,C=C)
            Eemp[irep,:] = fit_svm_dskl_emp(X,Y,Xtest,Ytest,nExpandSamples=cond,its=its,nPredSamples=nPred,eta=eta,C=C)
            Erks[irep,:] = fit_svm_dskl_rks(X,Y,Xtest,Ytest,nExpandSamples=cond,its=its,nPredSamples=nPred,eta=eta,C=C)
            Ebatch[irep,:] = fit_svm_batch(X,Y,Xtest,Ytest,1.)

        pl.clf()
        pl.plot(Erks.mean(axis=0),'k-')
        pl.plot(Eemp.mean(axis=0),'k.')
        pl.plot(EempFix.mean(axis=0),'k--')
        leg.append("Rks")
        leg.append("Emp")
        leg.append("Emp$_{Fix}$")
        pl.plot(Ebatch.mean()*sp.ones(Eemp.shape[1]),'k:')
        leg.append("Batch")
        if idx==0:pl.legend(leg,loc=3)
        pl.title("nExpand=%d"%cond)
        pl.xlabel("Iterations") 
        pl.ylabel("Error")
        pl.axis('tight')
        pl.ylim((0,.55))
        pl.savefig("rks_emp_comparison-pred-%d-expand-%d.pdf"%(nPred,cond))

def fit_svm_batch(X,Y,Xtest,Ytest,gamma):
    batchsvm = svm.SVC(gamma=gamma)
    batchsvm.fit(X.T,Y)
    return sp.mean(Ytest!=batchsvm.predict(Xtest.T))


def fit_svm_dskl_emp(X,Y,Xtest,Ytest,its=100,eta=1.,C=.1,nPredSamples=10,nExpandSamples=10, kernel=(GaussianKernel,(1.))):
    Wemp = sp.randn(len(Y))
    Eemp = []

    for it in range(1,its+1):
        Wempold = Wemp
        Wemp = step_dskl_empirical(X,Y,Wemp,eta/it,C,kernel,nPredSamples,nExpandSamples)
        diffW = Wempold.T.dot(Wemp)
        # for toy data sets we evaluate on all samples (but train on a small subset only)
        evaluateOn = range(len(Y))
        #evaluateOn = sp.random.randint(low=0,high=X.shape[1],size=nPredSamples)
        Eemp.append(sp.mean(Ytest[evaluateOn] != sp.sign(predict_svm_emp(X[:,evaluateOn],Xtest[:,evaluateOn],Wemp[evaluateOn],kernel)))) 
        #print "Error: %0.2f - diff norm:  %f"%(Eemp[-1], diffW)

    return Eemp

def fit_svm_dskl_rks(X,Y,Xtest,Ytest,its=100,eta=1.,C=.1,nPredSamples=10,nExpandSamples=10, kernel=(GaussianKernel,(1.))):
    # random gaussian for rks
    Zrks = sp.randn(len(Y),X.shape[0]) / (kernel[1]**2)
    Wrks = sp.randn(len(Y))

    Erks = []

    for it in range(1,its+1):
        Wrks = step_dskl_rks(X,Y,Wrks,Zrks,eta/it,C,nPredSamples,nExpandSamples)
        Erks.append(sp.mean(Ytest != sp.sign(predict_svm_rks(Xtest,Wrks,Zrks))))
        print "Error: %0.2f"%Erks[-1]
    return Erks

def fit_svm_dskl_comparison(X,Y,its=100,eta=1.,C=.1,nPredSamples=10,nExpandSamples=10, kernel=(GaussianKernel,(1.))):
    
    # split into train and test
    Xtest = X[:,:len(Y)/2]
    Ytest = Y[:len(Y)/2]
    X = X[:,(len(Y)/2):]
    Y = Y[(len(Y)/2):]

    # random gaussian for rks
    Zrks = sp.randn(len(Y),X.shape[0]) / (kernel[1]**2)
    Wrks = sp.randn(len(Y))

    Wemp = sp.randn(len(Y))

    Erks = []
    Eemp = []

    for it in range(1,its+1):
        Wrks = step_dskl_rks(X,Y,Wrks,Zrks,eta/it,C,nPredSamples,nExpandSamples)
        Wemp = step_dskl_empirical(X,Y,Wemp,eta/it,C,kernel,nPredSamples,nExpandSamples)
        Erks.append(sp.mean(Ytest != sp.sign(predict_svm_rks(Xtest,Wrks,Zrks))))
        Eemp.append(sp.mean(Ytest != sp.sign(predict_svm_emp(X,Xtest,Wemp,kernel)))) 
    return Eemp,Erks

def step_dskl_rks(X,Y,W,Z,eta=1.,C=.1,nPredSamples=10,nExpandSamples=10):
    rnpred = sp.random.randint(low=0,high=X.shape[1],size=nPredSamples)
    rnexpand = sp.random.randint(low=0,high=W.shape[0],size=nExpandSamples)
    if sp.sparse.issparse(X): X = sp.array(X.todense())
    # compute rks features
    rks_feats = sp.exp(Z[rnexpand,:].dot(X[:,rnpred])) / sp.sqrt(nPredSamples)
    # compute prediction
    yhat = W[rnexpand].T.dot(rks_feats)
    # compute whether or not prediction is in margin
    inmargin = (yhat * Y[rnpred]) <= 1
    # compute gradient for 
    G = C * W[rnexpand] - rks_feats.dot(Y[rnpred] * inmargin)
    W[rnexpand] -= eta * G
    return W

def predict_svm_rks(X,W,Z):
    if sp.sparse.issparse(X): X = sp.array(X.todense())
    return W.T.dot(sp.exp(Z.dot(X))/sp.sqrt(X.shape[1]))

def step_dskl_empirical(X,Y,W,eta=1.,C=.1,kernel=(GaussianKernel,(1.)),nPredSamples=10,nExpandSamples=10):
    if nPredSamples==0: rnpred=range(len(Y))
    else: rnpred = sp.random.randint(low=0,high=X.shape[1],size=nPredSamples)
    if nExpandSamples==0:rnexpand=range(len(Y))
    else: rnexpand = sp.random.randint(low=0,high=X.shape[1],size=nExpandSamples)
    # compute gradient 
    G = compute_gradient(Y[rnpred],X[:,rnpred],X[:,rnexpand],W[rnexpand],kernel,C)
    # update 
    W[rnexpand] -= eta * G
    return W

def compute_gradient(y,Xpred,Xexpand,w,kernel,C):
    if sp.sparse.issparse(Xpred): Xpred = sp.array(Xpred.todense())
    if sp.sparse.issparse(Xexpand): Xexpand = sp.array(Xexpand.todense())
    # compute kernel for random sample
    K = kernel[0](Xpred,Xexpand,kernel[1])
    # compute prediction 
    yhat = K.dot(w)
    # compute whether or not prediction is in margin
    inmargin = (yhat * y) <= 1
    # compute gradient for 
    G = C * w - (y * inmargin).dot(K)
    return G

def predict_svm_emp(Xexpand,Xtarget,w,kernel):
	return w.dot(kernel[0](Xexpand,Xtarget,kernel[1]))

def run_comparison_expand_krr(N=100,noise=0.1,nExpandSamples=[1,10,100],its=100,reps=100):
    pl.ion()
    pl.figure(figsize=(20,12))
    colors = "brymcwg"
    leg = []
    for idx,cond in enumerate(nExpandSamples):
        Eemp, Erks, Ebatch, EempFix = sp.zeros((reps,its)), sp.zeros((reps,its)), sp.zeros((reps,its)), sp.zeros((reps,its))
        for irep in range(reps):
            X,Y = make_data_cos(N*2,noise=noise)
            Y = Y.flatten()
            # split into train and test
            Xtest = X[:,:len(Y)/2]
            Ytest = Y[:len(Y)/2]
            X = X[:,(len(Y)/2):]
            Y = Y[(len(Y)/2):]

            EempFix[irep,:] = fit_krr_dskl_emp(X[:,:cond],Y[:cond],Xtest,Ytest,nExpandSamples=0,its=its)
            Eemp[irep,:] = fit_krr_dskl_emp(X,Y,Xtest,Ytest,nExpandSamples=cond,its=its)
            Erks[irep,:] = fit_krr_dskl_rks(X,Y,Xtest,Ytest,nExpandSamples=cond,its=its)

        pl.plot(Eemp.mean(axis=0),colors[idx]+'.')
        pl.plot(Erks.mean(axis=0),colors[idx]+'-')
        pl.plot(EempFix.mean(axis=0),colors[idx]+'--')
        leg.append("Emp, nSamp=%d"%cond)
        leg.append("Rks, nSamp=%d"%cond)
        leg.append("EmpFix, nSamp=%d"%cond)
    
    Ebatch = fit_krr_batch(X,Y,Xtest,Ytest)
    pl.plot(Ebatch.mean()*sp.ones(Eemp.shape[1]),color='black')
    leg.append("Batch, nSamp=%d"%N)
    pl.legend(leg)
    #pl.ylim(0,1.3)
    #pl.axis('tight')
    pl.savefig("rks_emp_comparison.pdf",)

def predict_krr_emp(X,Xtest,W,kernel=(GaussianKernel,(1.))): return W.dot(kernel[0](X,Xtest,kernel[1]))

def predict_krr_rks(Xtest,W,Z): return W.dot(sp.exp(Z.dot(Xtest)) / sp.sqrt(Xtest.shape[1]))

def error_krr(y,yhat): return sp.corrcoef(y,yhat)[0,1]#sp.mean((y-yhat)**2)

def fit_krr_batch_result(X,Y,Xtest,Ytest,kernel=(GaussianKernel,(1.))):
    return (sp.linalg.inv(kernel[0](X,X,kernel[1]) + sp.eye(X.shape[1]) * 1e-5).dot(Y)).dot(kernel[0](X,Xtest,kernel[1]))
    
def fit_krr_batch(X,Y,Xtest,Ytest,kernel=(GaussianKernel,(1.))):
    y = (sp.linalg.inv(kernel[0](X,X,kernel[1]) + sp.eye(X.shape[1]) * 1e-5).dot(Y)).dot(kernel[0](X,Xtest,kernel[1]))
    return error_krr(Ytest,y)

def fit_krr_dskl_emp_result(X,Y,Xtest,Ytest,its=100,eta=.1,C=.001,nPredSamples=10,nExpandSamples=10, kernel=(GaussianKernel,(1.))):
    Wemp = sp.randn(len(Y))
    Eemp = []

    for it in range(1,its+1):
        Wemp = step_dskl_empirical_krr(X,Y,Wemp,eta/it,C,kernel,nPredSamples,nExpandSamples)
    return predict_krr_emp(X,Xtest,Wemp,kernel)

def fit_krr_dskl_emp(X,Y,Xtest,Ytest,its=100,eta=.1,C=.001,nPredSamples=30,nExpandSamples=10, kernel=(GaussianKernel,(1.))):
    Wemp = sp.randn(len(Y))
    Eemp = []
    for it in range(1,its+1):
        Wemp = step_dskl_empirical_krr(X,Y,Wemp,eta/it,C,kernel,nPredSamples,nExpandSamples)
        Eemp.append(error_krr(Ytest,predict_krr_emp(X,Xtest,Wemp,kernel))) 
    return Eemp

def fit_krr_dskl_rks_result(X,Y,Xtest,Ytest,its=100,eta=.1,C=.001,nPredSamples=30,nExpandSamples=10, kernel=(GaussianKernel,(1.))):
    # random gaussian for rks
    Zrks = sp.randn(len(Y),X.shape[0]) / (kernel[1]**2)
    Wrks = sp.randn(len(Y))
    for it in range(1,its+1):
        Wrks = step_dskl_rks_krr(X,Y,Wrks,Zrks,eta/it,C,nPredSamples,nExpandSamples)
    return predict_krr_rks(Xtest,Wrks,Zrks) 


def fit_krr_dskl_rks(X,Y,Xtest,Ytest,its=100,eta=.1,C=.001,nPredSamples=30,nExpandSamples=10, kernel=(GaussianKernel,(1.))):
    # random gaussian for rks
    Zrks = sp.randn(len(Y),X.shape[0]) / (kernel[1]**2)
    Wrks = sp.randn(len(Y))
    Erks = []
    for it in range(1,its+1):
        Wrks = step_dskl_rks_krr(X,Y,Wrks,Zrks,eta/it,C,nPredSamples,nExpandSamples)
        Erks.append(error_krr(Ytest,predict_krr_rks(Xtest,Wrks,Zrks)))
    return Erks


def step_dskl_empirical_krr(X,Y,W,eta=1.,C=.1,kernel=(GaussianKernel,(1.)),nPredSamples=10,nExpandSamples=10):
    if nPredSamples==0: rnpred=range(len(Y))
    else: rnpred = sp.random.randint(low=0,high=X.shape[1],size=nPredSamples)
    if nExpandSamples==0:rnexpand=range(len(Y))
    else: rnexpand = sp.random.randint(low=0,high=X.shape[1],size=nExpandSamples)
    # compute gradient 
    G = compute_gradient_krr(Y[rnpred],X[:,rnpred],X[:,rnexpand],W[rnexpand],kernel,C)
    # update 
    W[rnexpand] -= eta * G
    return W

def step_dskl_rks_krr(X,Y,W,Z,eta=1.,C=.1,nPredSamples=10,nExpandSamples=10):
    rnpred = sp.random.randint(low=0,high=X.shape[1],size=nPredSamples)
    rnexpand = sp.random.randint(low=0,high=W.shape[0],size=nExpandSamples)
    # compute rks features
    rks_feats = sp.exp(Z[rnexpand,:].dot(X[:,rnpred])) / sp.sqrt(nPredSamples)
    # compute prediction
    yhat = W[rnexpand].T.dot(rks_feats)
    # compute gradient for 
    G = C * W[rnexpand] - rks_feats.dot(Y[rnpred]) + W[rnexpand].dot(rks_feats.dot(rks_feats.T))
    W[rnexpand] -= eta * G
    return W


def compute_gradient_krr(y,Xpred,Xexpand,w,kernel,C):
    # compute kernel for random sample
    K = kernel[0](Xpred,Xexpand,kernel[1])
    # compute prediction 
    yhat = K.dot(w)
    # compute gradient for 
    G = (C * w - y.dot(K) + w.dot(K.T.dot(K)))
    return G


def make_data_twoclass(N=50):
	# generates some toy data
	mu = sp.array([[0,2],[0,-2]]).T
	C = sp.array([[5.,4.],[4.,5.]])
	X = sp.hstack((mvn(mu[:,0],C,N/2).T, mvn(mu[:,1],C,N/2).T))
	Y = sp.hstack((sp.ones((1,N/2.)),-sp.ones((1,N/2.))))
	return X,Y
	
def make_data_xor(N=80,noise=.25):
    # generates some toy data
    mu = sp.array([[-1,1],[1,1]]).T
    C = sp.eye(2)*noise
    X = sp.hstack((mvn(mu[:,0],C,N/4).T,mvn(-mu[:,0],C,N/4).T, mvn(mu[:,1],C,N/4).T,mvn(-mu[:,1],C,N/4).T))
    Y = sp.hstack((sp.ones((1,N/2.)),-sp.ones((1,N/2.))))
    randidx = sp.random.permutation(N)
    Y = Y[0,randidx]
    X = X[:,randidx]
    return X,Y

    
def make_data_cos(N=100,noise=.3):
	# generates some toy data
	x = sp.randn(1,N)*sp.pi
	y = sp.cos(x) + sp.randn(1,N) * noise
	return x,y

def make_plot_twoclass(X,Y,W,kernel):
	fig = pl.figure(figsize=(5,4))
	fig.clf()
	colors = "brymcwg"

	# Plot the decision boundary.
	h = .2 # stepsize in mesh
	x_min, x_max = X[0,:].min() - 1, X[0,:].max() + 1
	y_min, y_max = X[1,:].min() - 1, X[1,:].max() + 1
	xx, yy = sp.meshgrid(sp.arange(x_min, x_max, h),
                     sp.arange(y_min, y_max, h))
                     
	Z = predict_svm_kernel(sp.c_[sp.ones(xx.ravel().shape[-1]), xx.ravel(), yy.ravel()].T,sp.vstack((sp.ones((1,X.shape[-1])),X)),W,kernel).reshape(xx.shape)
	cs = pl.contourf(xx, yy, Z,alpha=.5)
	pl.axis('tight')
	pl.colorbar()
	pl.axis('equal')
	y = sp.maximum(0,-Y)+1
	# plot the data
	pl.hold(True)

	ypred = 	W.T.dot(kernel[0](X,X,kernel[1]).T)
	for ic in sp.unique(y):
		idx = (y == int(ic)).flatten()
		sv = (Y.flatten()[idx]*ypred[idx] < 1)
		pl.plot(X[0,idx.nonzero()[0][sv]], X[1,idx.nonzero()[0][sv]], colors[int(ic)]+'o',markersize=13)
		pl.plot(X[0,idx.nonzero()[0][~sv]], X[1,idx.nonzero()[0][~sv]], colors[int(ic)]+'o',markersize=7)
	pl.axis('tight')

	pl.xlabel('$X_1$')
	pl.ylabel('$X_2$')

	#pl.title('SVM, Accuracy=%0.2f'%(Y==sp.sign(ypred)).mean())

	#pl.show()
	pl.savefig('./svm_kernel.pdf')

	fig = pl.figure(figsize=(5,5))
	fig.clf()
	colors = "brymcwg"
	for ic in sp.unique(y):
		idx = (y == int(ic)).flatten()
		pl.plot(X[0,idx], X[1,idx], colors[int(ic)]+'o',markersize=8)
	pl.axis('tight')

	pl.xlabel('$X_1$')
	pl.ylabel('$X_2$')
	pl.xlim((x_min,x_max))
	pl.ylim((y_min,y_max))
	pl.grid()
	#pl.show()
	pl.savefig('./svm_kernel_xor_data.pdf')
	
def plot_xor_example():
    k = GaussianKernel
    kparam = 1.
    N = 60
    noise = .2
    X,y = make_data_xor(N,noise)
    w = w = fit_svm_kernel(X,y.flatten())
    make_plot_twoclass(X,y,w.T,kernel=(k,(kparam)))


if __name__ == '__main__':
   plot_xor_example()
   run_comparison_expand()

