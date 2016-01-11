package Svm

import org.apache.commons.math3.linear.RealMatrix
import breeze.linalg._

import breeze.plot._

import scala.collection.mutable.ListBuffer

/**
  * Created by nikolaas steenbergen on 1/8/16.
  */
object Main {


  def main(args : Array[String]): Unit ={
    println("herro!")

//    var ret = Utils.make_data_xor(100,0.1);
//    var Xordered = ret._1
//    var Yordered = ret._2
//
//    // shuffle input data
//    var res0 = Utils.shuffleData(Xordered,Yordered)
//    var X = res0._1
//    var Y = res0._2
//
//    println("X:\n" + X)
//    println("Y: \n" + Y)
//
//    var N = X.cols
//
//    var W: DenseVector[Double] = DenseVector.rand(N)
//
//    var errors:DenseVector[Double] = null
//    var w: DenseVector[Double] = null
////    var res = UtilsDist.fit_svm_kernel(W, X, Y, iterations = 1000, eta = 1.0, C = 0.1)
//    var res = UtilsDist.fit_svm_kernel_flink(W, X, Y, iterations = 100, eta = 1.0, C = 0.1)
//
//    errors = res._2
//    w = res._1

//    var sigma: Double = 1.0
//
//    assert(errors != null && w != null,"check your privileges!")
//
//    Utils.plotData(X)
//    Utils.plotLine(errors)
//    Utils.plotModel(X, Y, w, sigma)
//    println("bye!")
    train_mnist()
  }

  def train_mnist(): Unit ={
    var res = UtilsDist.loadMnist()
    var X = res._1
    var Y_all = res._2

    var N = X.cols

    var Ys: ListBuffer[DenseMatrix[Double]] = UtilsDist.createOneVsAllTrainingsets(Y_all)

    var Y: DenseMatrix[Double] = Ys(0)
//    var res3 = UtilsDist.fit_svm_kernel_flink(W, X, Y, iterations = 100, eta = 1.0, C = 0.1)

    var ws: ListBuffer[DenseVector[Double]] = ListBuffer[DenseVector[Double]]()
    for (i <- 0 until 10){
      var W: DenseVector[Double] = DenseVector.rand(N)
      var res3 = Utils.fit_svm_kernel(W, X, Ys(i), iterations = 1000, eta = 1.0, C = 0.1)
      var errors = res3._2
      var w = res3._1
      ws += w
      Utils.plotLine(errors)
    }

    // run all the svms on the test data, write out confidence, vote
    for(point <- 0 until X.cols){
      for(svm <- 0 until 10){
        Utils.test_svm(X(::,point),)
      }
    }

  }

}
