import os
import time
import datetime


import scipy as sp
import numpy as np
import urllib
from scipy.sparse import csr_matrix

import sklearn
from sklearn.datasets import fetch_mldata
from sklearn.preprocessing import StandardScaler


custom_data_home = "/home/nikste/workspace-python/doubly_random_svm/"
if not os.path.isdir(custom_data_home):
    custom_data_home = "/home/mkaul/doubly_random_svm/"

mnist8mfn = "/home/nikste/workspace-python/doubly_random_svm/svmlightdata/infimnist/"
if not os.path.isdir(mnist8mfn):
    mnist8mfn = "/home/mkaul/doubly_random_svm/svmlightdata/infimnist/"

def load_clf(fname):
    f = file(fname,"rb")
    return np.pickle.load(f)



def scale_input(Xtrain, Xtest):
    print "scaling"
    print "current time:", datetime.datetime.now()
    t0 = time.time()

    Xtrain = csr_matrix((Xtrain.data, Xtrain.indices, Xtrain.indptr), shape=(Xtrain.shape[0], 784))
    Xtest = csr_matrix((Xtest.data, Xtest.indices, Xtest.indptr), shape=(Xtest.shape[0], 784))
    if not sp.sparse.issparse(Xtrain):
        scaler = StandardScaler()
        scaler.fit(Xtrain)  # Don't cheat - fit only on training data
        Xtrain = scaler.transform(Xtrain)
        Xtest = scaler.transform(Xtest)
    else:
        # standard scaler without mean scaling for sparse matrices (would break sparseness)
        if not Xtrain.shape[1] == Xtest.shape[1]:
            pass
        scaler = StandardScaler(with_mean=False)
        scaler.fit(Xtrain)  # Don't cheat - fit only on training data
        Xtrain = scaler.transform(Xtrain)
        Xtest = scaler.transform(Xtest)
    print "took:", time.time() - t0
    return Xtrain,Xtest




def get_svmlight_file(fn):
    from sklearn.datasets import load_svmlight_file
    url = "https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/"+fn
    fn = os.path.join(custom_data_home,"svmlightdata",fn)
    if not os.path.isfile(fn):
        print("Downloading %s to %s"%(url,fn))
        urllib.urlretrieve(url,fn)
    return load_svmlight_file(fn)


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


def load_mnist8m_distributed(only_zero_and_one = True):
    from sklearn.datasets import load_svmlight_file
    if only_zero_and_one:

        one_two_mnist8m_fn_train = mnist8mfn + "/mnist8m/" + "mnist8m-libsvm_0_1.txt"  # "mnist8m-libsvm_0_1_small.txt"
        one_two_mnist8m_fn_test = mnist8mfn + "/mnist8m/" + "mnist8m-libsvm_0_1-test.txt"  # "mnist8m-libsvm_0_1-test_small.txt"

        print "loading train"
        training_fns = ['aa','ab','ac','ad','ae','af']


        folder = tempfile.mkdtemp()
        data_name = os.path.join(folder, 'training_fns')
        dump(training_fns, data_name)
        training_fns_map = load(data_name, mmap_mode='r')



        t0 = time.time()
        dd = Parallel(n_jobs=8, backend="threading")(
            delayed(load_svmlight_file)(one_two_mnist8m_fn_train + training_fns[i]) for i in range(6))

        print "took:", time.time() - t0
        for el in dd:
            pass
        dd = load_svmlight_file(one_two_mnist8m_fn_train)
        Xtrain = dd[0]
        Ytrain = dd[1]
        Ytrain = sp.sign(Ytrain - .5)

        print "loading test"
        t0 = time.time()
        dd = load_svmlight_file(one_two_mnist8m_fn_test)
        Xtest = dd[0]
        Ytest = dd[1]
        Ytest = sp.sign(Ytest - .5)
        print "took:", time.time() - t0
        return Xtrain, Ytrain, Xtest, Ytest
    else:
        mnist8m_fn_train = mnist8mfn + "/mnist8m/mnist8m-libsvm.txt"
        mnist8m_fn_test = mnist8mfn + "/mnist8m/mnist8m-libsvm-test.txt"
        if not os.path.isfile(mnist8m_fn_train):
            raise ValueError("mnist8m train data not found in:" + mnist8m_fn_train)
        if not os.path.isfile(mnist8m_fn_test):
            raise ValueError("mnist8m test data not found in:" + mnist8m_fn_test)
        dd = load_svmlight_file(mnist8m_fn_train)
        Xtrain = dd[0]
        Ytrain = dd[1]
        Ytrain = sp.sign(Ytrain - .5)

        dd = load_svmlight_file(mnist8m_fn_test)
        Xtest = dd[0]
        Ytest = dd[1]
        Ytest = sp.sign(Ytest - .5)
        return Xtrain, Ytrain, Xtest, Ytest




def load_mnist8m(binary_classification = True):
    from sklearn.datasets import load_svmlight_file
    from sklearn.externals import joblib

    if binary_classification:
        binary_mnist8m_fn_train = mnist8mfn + "mnist8m/" + "mnist8m-libsvm_6_8_scaled.txt"#"mnist8m-libsvm_0_1_small.txt"
        binary_mnist8m_fn_test = mnist8mfn + "mnist8m/" + "mnist8m-libsvm_6_8_scaled-test.txt" #"mnist8m-libsvm_0_1-test_small.txt"


        print "loading train",datetime.datetime.now()
        t0 = time.time()
        Xtrain,Ytrain = joblib.load(binary_mnist8m_fn_train + ".dump")

        print "Xtrainshape=",Xtrain.shape, "Ytrainshape=", Ytrain.shape

        print "took:",time.time() - t0

        print "loading test"
        Xtest,Ytest = joblib.load(binary_mnist8m_fn_test + ".dump")
        print "Xtestshape=", Xtest.shape, "Ytestshape=", Ytest.shape
        print "took:", time.time() - t0
        return Xtrain,Ytrain,Xtest,Ytest

    else:
        mnist8m_fn_train = mnist8mfn + "/mnist8m/mnist8m-libsvm.txt"
        mnist8m_fn_test = mnist8mfn + "/mnist8m/mnist8m-libsvm-test.txt"
        if not os.path.isfile(mnist8m_fn_train):
            raise ValueError("mnist8m train data not found in:" + mnist8m_fn_train)
        if not os.path.isfile(mnist8m_fn_test):
            raise ValueError("mnist8m test data not found in:" + mnist8m_fn_test)
        dd = load_svmlight_file(mnist8m_fn_train)
        Xtrain = dd[0]
        Ytrain = dd[1]
        Ytrain = sp.sign(Ytrain - .5)

        dd = load_svmlight_file(mnist8m_fn_test)
        Xtest = dd[0]
        Ytest = dd[1]
        Ytest = sp.sign(Ytest - .5)
        return Xtrain, Ytrain, Xtest, Ytest



def preprocess_mnist8m():
    from sklearn.datasets import load_svmlight_file
    from sklearn.externals import joblib
    # preprocessing
    from sklearn.datasets import load_svmlight_file
    one_two_mnist8m_fn_train = "/home/nikste/workspace-python/doubly_random_svm_exp/svmlightdata/infimnist/" + "mnist8m/" + "mnist8m-libsvm_6_8.txt"   # "mnist8m-libsvm_0_1_small.txt"
    one_two_mnist8m_fn_test = "/home/nikste/workspace-python/doubly_random_svm_exp/svmlightdata/infimnist/" + "mnist8m/" + "mnist8m-libsvm_6_8-test.txt"  # "mnist8m-libsvm_0_1-test_small.txt"

    if not os.path.isfile(one_two_mnist8m_fn_train):
        raise ValueError("mnist8m train data not found in:" + one_two_mnist8m_fn_train)
    if not os.path.isfile(one_two_mnist8m_fn_test):
        raise ValueError("mnist8m test data not found in:" + one_two_mnist8m_fn_test)
    print "loading train"
    t0 = time.time()
    dd = load_svmlight_file(one_two_mnist8m_fn_train)
    Xtrain = dd[0]
    Ytrain = dd[1]
    Ytrain = sp.sign(Ytrain - .5)
    print "took:", time.time() - t0

    print "loading test"
    t0 = time.time()
    dd = load_svmlight_file(one_two_mnist8m_fn_test)
    Xtest = dd[0]
    Ytest = dd[1]
    Ytest = sp.sign(Ytest - .5)
    print "took:", time.time() - t0

    Xtrain,Xtest = scale_input(Xtrain, Xtest)


    print "saving to file:",datetime.datetime.now()
    mnist8mfn = "/home/nikste/workspace-python/doubly_random_svm_exp/svmlightdata/infimnist/mnist8m/"
    # sklearn.datasets.dump_svmlight_file(Xtrain, Ytrain, mnist8mfn + "mnist8m-libsvm_0_1_scaled.txt")
    # sklearn.datasets.dump_svmlight_file(Xtest, Ytest, mnist8mfn + "mnist8m-libsvm_0_1-test_scaled.txt")


    joblib.dump((Xtrain, Ytrain), mnist8mfn  + "mnist8m-libsvm_6_8_scaled.txt.dump", cache_size=200, protocol=2)
    joblib.dump((Xtest, Ytest), mnist8mfn + "mnist8m-libsvm_6_8_scaled-test.txt.dump" , cache_size=200, protocol=2)