import datetime
import pickle

import scipy as sp
import numpy as np
from scipy.sparse import csr_matrix

from sklearn import svm
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import StandardScaler


from dataio import load_realdata, custom_data_home
from dsekl import DSEKL, DSEKLBATCH


def run_all_realdata(dnames=['sonar','mushroom','skin_nonskin','covertype','diabetes','gisette']):
    [run_realdata(dname=d) for d in dnames]


def run_realdata_no_comparison(dname='sonar', n_its=1000, num_test=1000, worker=8, maxN=1000):
    print "started loading:", datetime.datetime.now()
    Xtotal, Ytotal = load_realdata(dname)

    print "loading data done!", datetime.datetime.now()
    # decrease dataset size
    N = Xtotal.shape[0]
    if maxN > 0:
        N = sp.minimum(Xtotal.shape[0], maxN)

    if(N + num_test > Xtotal.shape[0]):
        print "number of training and testing samles is greater than original data:", N + num_test , " > ",Xtotal.shape[0]
        print "resetting training samples to N = ",Xtotal.shape[0] - num_test
        N = Xtotal.shape[0] - num_test

    # Xtotal = Xtotal[:N + num_test]
    # Ytotal = Ytotal[:N + num_test]

    # randomize datapoints
    print "randomization", datetime.datetime.now()
    sp.random.seed(0)
    idx = sp.random.permutation(Xtotal.shape[0])
    print idx
    Xtotal = Xtotal[idx]
    Ytotal = Ytotal[idx]

    # divide test and train
    print "dividing in train and test", datetime.datetime.now()
    Xtest = Xtotal[-num_test:]
    Ytest = Ytotal[-num_test:]
    Xtrain = Xtotal[0:N]
    Ytrain = Ytotal[0:N]

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
    DS = DSEKLBATCH(n_pred_samples=10000,n_expand_samples=10000,eta=0.999,n_its=n_its,C=1./float(N),gamma=1.0,damp=True,workers=worker,validation=True,verbose=True).fit(Xtrain,Ytrain)
    pickle.dump(DS,file("DSBatch_nodense","wb"),pickle.HIGHEST_PROTOCOL)

    print "test result subsample:", sp.mean(sp.sign(DS.predict(Xtest))!=Ytest)

def run_realdata_no_comparison_scaleup_experiments(dname='sonar', n_its=1000, num_test=1000, worker=8, maxN=1000):
    print "started loading:", datetime.datetime.now()
    Xtotal, Ytotal = load_realdata(dname)

    print "loading data done!", datetime.datetime.now()
    # decrease dataset size
    N = Xtotal.shape[0]
    if maxN > 0:
        N = sp.minimum(Xtotal.shape[0], maxN)

    if(N + num_test > Xtotal.shape[0]):
        print "number of training and testing samles is greater than original data:", N + num_test , " > ",Xtotal.shape[0]
        print "resetting training samples to N = ",Xtotal.shape[0] - num_test
        N = Xtotal.shape[0] - num_test

    # Xtotal = Xtotal[:N + num_test]
    # Ytotal = Ytotal[:N + num_test]

    # randomize datapoints
    print "randomization", datetime.datetime.now()
    sp.random.seed(0)
    idx = sp.random.permutation(Xtotal.shape[0])
    print idx
    Xtotal = Xtotal[idx]
    Ytotal = Ytotal[idx]

    # divide test and train
    print "dividing in train and test", datetime.datetime.now()
    Xtest = Xtotal[-num_test:]
    Ytest = Ytotal[-num_test:]
    Xtrain = Xtotal[0:N]
    Ytrain = Ytotal[0:N]

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


    # Xtrain_large = np.zeros((Xtrain.shape[0] * 2, Xtrain.shape[1]))
    # Ytrain_large = np.zeros(Ytrain.shape[0]*2)
    # s0 = Xtrain.shape[0]
    # Xtrain_large[0:s0,:] = Xtrain.copy()
    # Ytrain_large[0:s0] = Ytrain.copy()

    # Xtrain_large[s0:s0*2,:] = Xtrain.copy()
    # Ytrain_large[s0:s0*2] = Ytrain.copy()

    # Xtrain_large[s0*2:s0*3,:] = Xtrain.copy()
    # Ytrain_large[s0*2:s0*3] = Ytrain.copy()

    # Xtrain_large[s0*3:s0*4,:] = Xtrain.copy()
    # Ytrain_large[s0*3:s0*4] = Ytrain.copy()
    # Xtrain = Xtrain_large
    # Ytrain = Ytrain_large
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
    time_differences = []
    for i in range(48,1,-2):
        t_start = datetime.datetime.now()
        #print "starting training process:",i," time: ",t_start
        # max: n_pred_samples=15000,n_expand_samples=15000
        DS = DSEKLBATCH(n_pred_samples=20000,n_expand_samples=20000,eta=0.999,n_its=n_its,C=1./float(N),gamma=1.0,damp=True,workers=worker,validation=False,verbose=True,benefit_through_distribution_no_workers=i).fit(Xtrain,Ytrain)
        t_diff = datetime.datetime.now() - t_start
        print "stopping training process",i," time: ",datetime.datetime.now(), " time difference: ", t_diff
        time_differences.append((i,t_diff))
    import csv


    with open('scaling_experiments_8cores.csv', 'w') as out:
        csv_out = csv.writer(out)
        csv_out.writerow(['number_workers', 'time'])
        for row in time_differences:
            csv_out.writerow(row)

def dsekl_test_predict(dname='sonar', num_test=1000, maxN=1000):
    print "started loading:", datetime.datetime.now()
    Xtotal, Ytotal = load_realdata(dname)

    print "loading data done!", datetime.datetime.now()
    # decrease dataset size
    N = Xtotal.shape[0]
    if maxN > 0:
        N = sp.minimum(Xtotal.shape[0], maxN)

    Xtotal = Xtotal[:N + num_test]
    Ytotal = Ytotal[:N + num_test]

    # randomize datapoints
    print "randomization", datetime.datetime.now()
    sp.random.seed(0)
    idx = sp.random.permutation(Xtotal.shape[0])
    print idx
    Xtotal = Xtotal[idx]
    Ytotal = Ytotal[idx]

    # divide test and train
    print "dividing in train and test", datetime.datetime.now()
    Xtest = Xtotal[N:N+num_test]
    Ytest = Ytotal[N:N+num_test]
    Xtrain = Xtotal[:N]
    Ytrain = Ytotal[:N]

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

    DS = pickle.load(file("DS","rb"))

    res_hl = DS.predict_support_hardlimits(Xtest)
    res_hl = sp.mean(sp.sign(res_hl) != Ytest)
    print "res_hl",res_hl
    res_perc = DS.predict_support_percentiles(Xtest)
    res_perc = sp.mean(sp.sign(res_perc) != Ytest)
    print "res_perc",res_perc




def hyperparameter_search_dskl(reps=2,dname='sonar',maxN=1000,num_test=10000):
    Xtotal, Ytotal = load_realdata(dname)

    if maxN > 0:
        N = sp.minimum(Xtotal.shape[0], maxN)
    else:
        N = Xtotal.shape[0]

    params_dksl = {
        'n_pred_samples': [300],#[int(N/2*0.01)],#,int(N/2*0.02),int(N/2*0.03)],
        'n_expand_samples': [300],#[int(N/2*0.01)],#,int(N/2*0.02),int(N/2*0.03)],
        'n_its': [10000],
        'eta': [0.999],
        'C': 10. ** sp.arange(-30., -19., 1.),#2
        'gamma': [10.], #** sp.arange(0., 4., 1.),#2
        'workers': [7],
    }

    print "checking parameters:\n",params_dksl

    Eemp = []

    for irep in range(reps):
        print "repetition:",irep," of ",reps
        #idx = sp.random.randint(low=0,high=Xtotal.shape[0],size=N + num_test)
        Xtrain = Xtotal[:N,:]#idx[:N],:]
        Ytrain = Ytotal[:N]#idx[:N]]
        # TODO: check when if we have enough data here.
        Xtest = Xtotal[N:N+num_test,:]#idx[N:N+num_test],:]
        Ytest = Ytotal[N:N+num_test]#idx[N:N+num_test]]

        # Xtrain = Xtrain.todense()
        # Xtest = Xtest.todense()
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
        # clf = GridSearchCV(DSEKL(),params_dksl,n_jobs=-1,verbose=1,cv=2).fit(Xtrain,Ytrain)
        clf = GridSearchCV(DSEKLBATCH(),params_dksl,n_jobs=-1,verbose=1,cv=2).fit(Xtrain,Ytrain)
        print clf.best_estimator_.get_params()
        Eemp.append(sp.mean(sp.sign(clf.best_estimator_.transform(Xtest))!=Ytest))
        print "Emp: %0.2f"%(Eemp[-1])
        fname = custom_data_home + "clf_scmallscale_" + dname + "_nt" + str(N) + "_" + str(irep) + str(datetime.datetime.now())
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

        # Xtrain = Xtrain.todense()
        # Xtest = Xtest.todense()
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
        clf = GridSearchCV(DSEKL(),params_dksl,n_jobs=10,verbose=1,cv=2).fit(Xtrain,Ytrain)
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
    '''
    IMPORTANT if does not use all cores try start with:
    Shell>OPENBLAS_MAIN_FREE=1 python myscript.py
    '''
    print "started",datetime.datetime.now()
    # dsekl_test_predict(dname='covertype',maxN=1000,num_test=10000)
    # run_realdata(reps=10, dname='covertype', maxN=2000)
    # hyperparameter_search_dskl(reps=2,dname="mnist8m_crossvalidation",maxN=5000,num_test=1000)

    # run_realdata_no_comparison(dname='covertype',n_its=20000,worker=48,maxN=-1,num_test=10000)
    #run_realdata_no_comparison(dname='covertype',n_its=10000,worker=-1,maxN=-1,num_test=20000)
    # run_realdata_no_comparison_scaleup_experiments(dname='covertype',n_its=1,worker=-1,maxN=-1,num_test=20000)
    run_realdata_no_comparison_scaleup_experiments(dname='covertype',n_its=1,worker=-1,maxN=100000*48,num_test=10000)

