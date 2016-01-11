package Svm

import breeze.linalg.{max, DenseVector, DenseMatrix}
import org.apache.flink.api.common.functions.{ReduceFunction, MapFunction, FlatMapFunction}
import org.apache.flink.api.scala._

import org.apache.flink
import org.apache.flink.ml.common.LabeledVector


import org.apache.flink.api.scala._
import org.apache.flink.ml.math.Vector
import org.apache.flink.ml.common.LabeledVector
import org.apache.flink.ml.RichExecutionEnvironment

import scala.collection.mutable.ListBuffer

/**
  * Created by nikolaas steenbergen on 1/10/16.
  */
object UtilsDist {


  var env = ExecutionEnvironment.getExecutionEnvironment

  def loadMnist():(DenseMatrix[Double],DenseMatrix[Double]) = {

    var filepath = "/media/owner/extension/mnist_m8/infimnist/mnist_std/test-libsvm"
    // Read the training data set, from a LibSVM formatted file
    val trainingDS: DataSet[LabeledVector] = env.readLibSVM(filepath)

    val trainingcollect = trainingDS.collect()

    var trainingdata = DenseMatrix.zeros[Double](trainingcollect(0).vector.size + 1 ,trainingcollect.size)

    //TODO: conversion can be done with flink as well
    for(i <- 0 until trainingcollect.size){
      if(i % 100 == 0){println(i)}
      trainingdata(0,i) = trainingcollect(i).label
      var flinkVec: Vector = trainingcollect(i).vector
      for(j <- 0 until flinkVec.size){
        var flinkvecval = flinkVec(j)
        trainingdata(j + 1,i) = flinkvecval
      }
    }

    var trainingdatamat = trainingdata

    var N: Int = trainingdata.cols
    var D: Int = trainingdatamat.rows - 1
    var X: DenseMatrix[Double] = DenseMatrix.zeros[Double](D,N)
    var Y: DenseMatrix[Double] = DenseMatrix.zeros[Double](1,N)

    for(i <- 0 until trainingdatamat.cols){
      Y(0,i) = trainingdatamat(0,i)
      X(::,i) := trainingdatamat(1 until trainingdatamat.rows,i)
    }
    return (X,Y)
  }


  def createOneVsAllTrainingsets(Y: DenseMatrix[Double]): ListBuffer[DenseMatrix[Double]] ={
    // 10 classes

    // should be int anyway
    var num_classes: Int = max(Y).toInt + 1
    var trainingDataAllClasses: ListBuffer[DenseMatrix[Double]] = ListBuffer[DenseMatrix[Double]]()

    for( classIdx <- 0 until num_classes){
      var Y_current: DenseMatrix[Double] = DenseMatrix.zeros[Double](1,Y.cols)
      for( col <- 0 until Y.cols){
        if(Y(::,col) != classIdx){
          Y_current(0,col) = -1
        }else{
          Y_current(0,col) = 1
        }
      }
      trainingDataAllClasses += Y_current
    }
    return trainingDataAllClasses
  }
  /**
    * flink job
    * @param W
    * @param X_org
    * @param Y
    * @param iterations
    * @param eta
    * @param C
    * @param sigma
    * @return
    */
  def fit_svm_kernel_flink(W: DenseVector[Double], X_org: DenseMatrix[Double], Y: DenseMatrix[Double], iterations: Int, eta: Double = 1.0, C: Double = 0.1, sigma:Double = 1.0): (DenseVector[Double], DenseVector[Double]) = {

    var D: Int = X_org.rows
    var N: Int = X_org.cols
    var X = DenseMatrix.vertcat[Double](DenseMatrix.ones[Double](1, N), X_org)

    // split up to number of partitions:
    var partitions: Int = 10

    assert(N % partitions == 0)
    var stepsize: Int = N / partitions
    var errorsAll: DenseVector[Double] = DenseVector.zeros[Double](iterations * partitions)


    var dataArray: ListBuffer[(Int,DenseMatrix[Double],DenseMatrix[Double],DenseVector[Double])] = ListBuffer[(Int,DenseMatrix[Double],DenseMatrix[Double],DenseVector[Double])]()
    //TODO: this should be done differently to prevent filling up the memory with all the data!
    for(i <- 1 until partitions){
      var mini: Int = i * stepsize
      var maxi: Int = (i + 1) * stepsize
      dataArray += ((i,X(::,mini until maxi),Y(::,mini until maxi),W(mini until maxi)))
    }
    for(el <- dataArray){
      println(el)
    }

    println("length:" + dataArray)
    var dataSet = env.fromCollection[(Int,DenseMatrix[Double],DenseMatrix[Double],DenseVector[Double])](dataArray)

    var result = dataSet.map(new MapFunction[(Int,DenseMatrix[Double],DenseMatrix[Double],DenseVector[Double]),(Int,DenseVector[Double],DenseVector[Double])] {
      override def map(t: (Int, DenseMatrix[Double], DenseMatrix[Double], DenseVector[Double])): (Int, DenseVector[Double], DenseVector[Double]) = {
        var key = t._1
        var Xlocal = t._2
        var Ylocal = t._3
        var Wlocal = t._4

        var res = fit_svm_kernel_one_node(Wlocal, Xlocal, Ylocal, iterations, eta = eta, C = C, sigma = sigma)

        var ret = (key,res._1,res._2)
        println(ret)
        ret
      }
    })
//
    var collected = result.collect()

    for(el <- collected){
      var mini: Int = el._1 * stepsize
      var maxi: Int = (el._1 + 1) * stepsize
      W(mini until maxi) := el._2
      var minErr: Int = el._1 * iterations
      var maxErr: Int = (el._1 + 1) * iterations
      errorsAll( minErr until maxErr) := el._3
    }
    return (W,errorsAll)
  }




  def fit_svm_kernel(W: DenseVector[Double], X_org: DenseMatrix[Double], Y: DenseMatrix[Double], iterations: Int, eta: Double = 1.0, C: Double = 0.1, sigma:Double = 1.0): (DenseVector[Double], DenseVector[Double]) = {
    var D: Int = X_org.rows
    var N: Int = X_org.cols
    var X = DenseMatrix.vertcat[Double](DenseMatrix.ones[Double](1, N), X_org)

    // split up to number of partitions:
    var partitions: Int = 10

    assert(N % partitions == 0, "number of datapoints " + N + " should be divisible by number of partitions " + partitions)
    var stepsize: Int = N / partitions
    var errorsAll: DenseVector[Double] = DenseVector.zeros[Double](iterations * partitions)

    //TODO: parallelize here
    for(i <- 0 until partitions){
      var mini: Int = i * stepsize
      var maxi: Int = (i + 1) * stepsize

      var Ylocal: DenseMatrix[Double] = Y(::,mini until maxi)
      var Wlocal: DenseVector[Double] = W(mini until maxi)
      var Xlocal: DenseMatrix[Double] = X(::, mini until maxi)

      var W_sub: DenseVector[Double] = null
      var errors: DenseVector[Double] = null
      var res = fit_svm_kernel_one_node(Wlocal,Xlocal,Ylocal,iterations,eta = eta,C = C,sigma = sigma)

      W_sub = res._1
      errors = res._2

      W(mini until maxi) := W_sub
      var minErr: Int = i * iterations
      var maxErr: Int = (i + 1) * iterations
      errorsAll( minErr until maxErr) := errors
    }
//    TODO: errors do not refelect real training error (on all datapoints)!
    return (W,errorsAll)
  }


  def fit_svm_kernel_one_node(W: DenseVector[Double], X: DenseMatrix[Double], Y: DenseMatrix[Double], iterations: Int, eta: Double = 1.0, C: Double = 0.1, sigma:Double = 1.0): (DenseVector[Double],DenseVector[Double]) = {
    var D: Int = X.rows
    var N: Int = X.cols


    var errors:DenseVector[Double] = DenseVector.zeros[Double](iterations);
    for(i <- 0 until iterations){
      var err = Utils.test_svm(X,Y,W,sigma)
      errors(i) = err
      println("iteration:" + i + "erorr:" + errors(i))
      //var rn = Random.nextInt(N)
      var rn = i%N
      var yhat: Double = Utils.predict_svm_kernel_all(X(::,rn),X,W,sigma)
      var discount = eta/(i + 1.)

      var r = yhat * Y(::,rn)
      assert(r.length == 1)

      var G: DenseVector[Double] = DenseVector.zeros[Double](N)
      if (r(0) > 1) {
        G = C * W
      } else {
        var Y_ = Y(::,rn)
        G = C * W - Y_(0) * Utils.gaussianKernel(X(::,rn),X,sigma)
      }
      W := W - discount * G
    }

    return (W,errors)
  }



}
