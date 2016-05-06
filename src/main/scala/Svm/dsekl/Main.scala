package Svm.dsekl


import Svm.dsekl.DSEKLBATCH;

import java.io.File

import breeze.linalg._

/**
  * Created by nikste on 5/4/16.
  */
object Main {

  var home_path = "/home/nikste/workspace-python/doubly_random_svm/data/covertype/"
  var home_path_server = "/data/nsteenbergen/covertype/"

  def main(args : Array[String]): Unit ={
    println("herro!")
    var use_flink = false
    var Xtrain: DenseMatrix[Double] = csvread(new File(home_path + "Xtrain_10000"),separator=',').toDenseMatrix;
    var Xtest: DenseMatrix[Double] = csvread(new File(home_path + "Xtest_10000"),separator=',').toDenseMatrix;
    var Ytest: DenseVector[Double] = csvread(new File(home_path + "Ytest_10000"),separator=',').toDenseVector;
    var Ytrain: DenseVector[Double] = csvread(new File(home_path + "Ytrain_10000"),separator=',').toDenseVector;


    var n_its = 10000
    var N = Xtrain.rows
    var worker = 8

    var DS = new DSEKLBATCH(n_pred_samples=4000,
      n_expand_samples=4000,
      eta=0.999,
      n_its=n_its,
      C=(1.0/N.toDouble),
      gamma=1.0,
      damp=true,
      workers=worker,
      validation=true,
      verbose=true).fit(Xtrain,Ytrain)//n_pred_samples=4000,n_expand_samples=4000,eta=0.999,n_its=n_its,C=1./float(N),gamma=1.0,damp=True,workers=worker,validation=True,verbose=True).fit(Xtrain,Ytrain)



    println("bye!")
  }
}
