import datetime
import pickle

import scipy as sp
from scipy.sparse import csr_matrix

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler


from dataio import load_realdata, custom_data_home
from dsekl import DSEKL



def run_all_realdata(dnames=['sonar','mushroom','skin_nonskin','covertype','diabetes','gisette']):
    [run_realdata(dname=d) for d in dnames]


def run_realdata_no_comparison(dname='sonar', n_its=1000, percent_train=0.9, worker=8, maxN=1000):
    print "started loading:", datetime.datetime.now()
    Xtotal, Ytotal = load_realdata(dname)

    print "loading data done!", datetime.datetime.now()
    # decrease dataset size
    N = Xtotal.shape[0]
    if maxN > 0:
        N = sp.minimum(Xtotal.shape[0], maxN)

    Xtotal = Xtotal[:N]
    Ytotal = Ytotal[:N]

    # randomize datapoints
    print "randomization", datetime.datetime.now()
    idx = sp.random.permutation(Xtotal.shape[0])
    Xtotal = Xtotal[idx]
    Ytotal = Ytotal[idx]

    # divide test and train
    n_train = int(N * percent_train)
    print "dividing in train and test", datetime.datetime.now()
    Xtest = Xtotal[n_train:]
    Ytest = Ytotal[n_train:]
    Xtrain = Xtotal[:n_train]
    Ytrain = Ytotal[:n_train]

    print "densifying", datetime.datetime.now()
    # unit variance and zero mean
    Xtrain = Xtrain.todense()
    Xtest = Xtest.todense()

    if not sp.sparse.issparse(Xtrain):
        scaler = StandardScaler()
        print "fitting scaler", datetime.datetime.now()
        scaler.fit(Xtrain)  # Don't cheat - fit only on training data
        print "transforming data train", datetime.datetime.now()
        Xtrain = scaler.transform(Xtrain)
        print "transforming data test", datetime.datetime.now()
        Xtest = scaler.transform(Xtest)
    else:
        scaler = StandardScaler(with_mean=False)
        scaler.fit(Xtrain)
        Xtrain = scaler.transform(Xtrain)
        Xtest = scaler.transform(Xtest)

    '''
    sonar:
    {'C': 0.0001, 'n_pred_samples': 100, 'workers': 1, 'n_expand_samples': 100, 'eta': 1.0, 'n_its': 1000, 'gamma': 1.0}
    {'kernel': 'rbf', 'C': 100.0, 'verbose': False, 'probability': False, 'degree': 3, 'shrinking': True, 'max_iter': -1, 'decision_function_shape': None, 'random_state': None, 'tol': 0.001, 'cache_size': 200, 'coef0': 0.0, 'gamma': 0.01, 'class_weight': None}
    '''

    '''
    Emp: 0.51 - Batch: 0.51
    {'C': 1.0, 'n_pred_samples': 100, 'workers': 1, 'n_expand_samples': 500, 'eta': 1.0, 'n_its': 1000, 'gamma': 100.0}
    {'kernel': 'rbf', 'C': 100.0, 'verbose': False, 'probability': False, 'degree': 3, 'shrinking': True, 'max_iter': -1, 'decision_function_shape': None, 'random_state': None, 'tol': 0.001, 'cache_size': 200, 'coef0': 0.0, 'gamma': 0.0001, 'class_weight': None}
    '''
    '''
    bigger smaple size:
    {'C': 9.9999999999999995e-07, 'n_pred_samples': 500, 'workers': 1, 'n_expand_samples': 1000, 'eta': 1.0, 'n_its': 1000, 'gamma': 1.0}
    '''
    print "starting training process",datetime.datetime.now()
    # max: n_pred_samples=15000,n_expand_samples=15000
    worker = 48
    DS = DSEKL(n_pred_samples=15000,n_expand_samples=15000,n_its=n_its,C=1e-08,gamma=1.0,workers=worker,validation=True,verbose=True).fit(Xtrain,Ytrain)
    # svm = svm.SVC(n_its=n_its,C=9.9999999999999995e-07,gamma=1.0)
    # print "test result all:", sp.mean(sp.sign(DS.predict_all(Xtest))!=Ytest)
    # print "smart subsample:", sp.mean(sp.sign(DS.predict_support(Xtest))!=Ytest)
    print "test result subsample:", sp.mean(sp.sign(DS.predict(Xtest))!=Ytest)


def hyperparameter_search_dskl(reps=2,dname='sonar',maxN=1000,num_test=10000):
    Xtotal, Ytotal = load_realdata(dname)

    if maxN > 0:
        N = sp.minimum(Xtotal.shape[0], maxN)
    else:
        N = Xtotal.shape[0]

    params_dksl = {
        'n_pred_samples': [N/2*0.01,N/2*0.02,N/2*0.03],
        'n_expand_samples': [N/2*0.01,N/2*0.02,N/2*0.03],
        'n_its': [10000],
        'eta': [1.],
        'C': 10.  **sp.arange(-8.,4.,2.),
        'gamma': 10. **sp.arange(-4.,4.,2.),
        'workers': [48],
        #'validation': [False],
        #'damp:': [True,False]#,
        #'verbose': [False]#,
    }

    print "checking parameters:\n",params_dksl

    Eemp = []

    for irep in range(reps):
        print "repetition:",irep," of ",reps
        idx = sp.random.randint(low=0,high=Xtotal.shape[0],size=N + num_test)
        Xtrain = Xtotal[idx[:N],:]
        Ytrain = Ytotal[idx[:N]]
        # TODO: check when if we have enough data here.
        Xtest = Xtotal[idx[N:N+num_test],:]
        Ytest = Ytotal[idx[N:N+num_test]]

        Xtrain = Xtrain.todense()
        Xtest = Xtest.todense()
        if not sp.sparse.issparse(Xtrain):
            scaler = StandardScaler()
            scaler.fit(Xtrain)  # Don't cheat - fit only on training data
            Xtrain = scaler.transform(Xtrain)
            Xtest = scaler.transform(Xtest)
        else:
            scaler = StandardScaler(with_mean=False)
            scaler.fit(Xtrain)
            Xtrain = scaler.transform(Xtrain)
            Xtest = scaler.transform(Xtest)
        print "Training empirical"
        clf = GridSearchCV(DSEKL(),params_dksl,n_jobs=-1,verbose=1,cv=2).fit(Xtrain,Ytrain)
        Eemp.append(sp.mean(sp.sign(clf.best_estimator_.transform(Xtest))!=Ytest))
        print "Emp: %0.2f"%(Eemp[-1])
        print clf.best_estimator_.get_params()
        fname = custom_data_home + "clf_" + dname + "_nt" + str(N) + "_reps_damp_True_its_10000_nodiscount" + str(irep) + datetime.datetime.now()
        f = open(fname,'wb')
        print "saving to file:", fname
        pickle.dump(clf, f, pickle.HIGHEST_PROTOCOL)
    print "***************************************************************"
    print "Data set [%s]: Emp_avg: %0.2f+-%0.2f"%(dname,sp.array(Eemp).mean(),sp.array(Eemp).std())
    print "***************************************************************"


def run_realdata(reps=2,dname='sonar',maxN=1000):
    Xtotal,Ytotal = load_realdata(dname)

    params_dksl = {
            'n_pred_samples': [500,1000],
            'n_expand_samples': [500,1000],
            'n_its':[1000],
            'eta':[1.],
            'C':10.**sp.arange(-8.,4.,2.),#**sp.arange(-8,-6,1),#[1e-6],#
            'gamma':10.**sp.arange(-4.,4.,2.),#**sp.arange(-1.,2.,1)#[10.]#
            'workers':[500,1000]
            }
    
    params_batch = {
            'C':10.**sp.arange(-8.,4.,2.),
            'gamma':10.**sp.arange(-4.,4.,2.)
            }
 
    if maxN > 0:
        N = sp.minimum(Xtotal.shape[0],maxN)
    else:
        N = Xtotal.shape[0]

    Eemp,Ebatch = [],[]
    num_train = int(0.9*N)
    for irep in range(reps):
        print "repetition:",irep," of ",reps
        idx = sp.random.randint(low=0,high=Xtotal.shape[0],size=N)
        Xtrain = Xtotal[idx[:num_train],:]
        Ytrain = Ytotal[idx[:num_train]]
        Xtest = Xtotal[idx[num_train:],:]
        Ytest = Ytotal[idx[num_train:]]

        Xtrain = Xtrain.todense()
        Xtest = Xtest.todense()
        if not sp.sparse.issparse(Xtrain):
            scaler = StandardScaler()
            scaler.fit(Xtrain)  # Don't cheat - fit only on training data
            Xtrain = scaler.transform(Xtrain)
            Xtest = scaler.transform(Xtest)
        else:
            scaler = StandardScaler(with_mean=False)
            scaler.fit(Xtrain)
            Xtrain = scaler.transform(Xtrain)
            Xtest = scaler.transform(Xtest)
        print "Training empirical"
        clf = GridSearchCV(DSEKL(),params_dksl,n_jobs=10,verbose=1,cv=3).fit(Xtrain,Ytrain)
        Eemp.append(sp.mean(sp.sign(clf.best_estimator_.transform(Xtest))!=Ytest))
        clf_batch = GridSearchCV(svm.SVC(),params_batch,n_jobs=1000,verbose=1,cv=3).fit(Xtrain,Ytrain)
        Ebatch.append(sp.mean(clf_batch.best_estimator_.predict(Xtest)!=Ytest))
        print "Emp: %0.2f - Batch: %0.2f"%(Eemp[-1],Ebatch[-1])
        print clf.best_estimator_.get_params()
        print clf_batch.best_estimator_.get_params()
    print "***************************************************************"
    print "Data set [%s]: Emp_avg: %0.2f+-%0.2f - Ebatch_avg: %0.2f+-%0.2f"%(dname,sp.array(Eemp).mean(),sp.array(Eemp).std(),sp.array(Ebatch).mean(),sp.array(Ebatch).std())
    print "***************************************************************"




if __name__ == '__main__':

    # run_realdata(reps=10, dname='covertype', maxN=2000)
    hyperparameter_search_dskl(reps=2,dname="covertype",maxN=10000)
    # run_realdata_no_comparison(dname='covertype',n_its=20000,worker=1,maxN=15000)

