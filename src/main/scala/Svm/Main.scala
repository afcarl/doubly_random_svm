package Svm

import org.apache.commons.math3.linear.RealMatrix
import breeze.linalg._

import breeze.plot._
/**
  * Created by owner on 1/8/16.
  */
object Main {



  def runSample(): Unit = {
    var X: DenseMatrix[Double] = DenseMatrix((1d,-1d,-1d,1d),(1d,-1d,1d,-1d))
    println("X:\n" + X)
    var Y: DenseMatrix[Double] = DenseMatrix(1d,1d,-1d,-1d).t
    println("Y:\n"+Y)
    var W: DenseVector[Double] = DenseVector(1d,1d,1d,1d)

    Utils.plotData(X)
    var (w,errors) = Utils.fit_svm_kernel(W, X, Y, iterations = 100, eta = 1.0, C = 0.001)
    var sigma = 1.0
    Utils.plotModel(X, w, sigma)
  }
  def main(args : Array[String]): Unit ={
    println("herro!")
//    runSample()

    var ret = Utils.make_data_xor(500,0.1);
    var X = ret._1
    var Y = ret._2

    println("X:\n" + X)
    println("Y: \n" + Y)

    var N = X.cols

    var W: DenseVector[Double] = DenseVector.rand(N)
//    var res = Utils.predict_svm_kernel_all(X(::,0),X, W, 1.0d)

//    println("res:\n" + res)



    var (w,errors) = Utils.fit_svm_kernel(W, X, Y, iterations = 1000, eta = 1.0, C = 0.1)

    var sigma: Double = 1.0

    Utils.plotData(X)
    Utils.plotLine(errors)
    Utils.plotModel(X, w, sigma)
//    Plotter.scatterPlot()
    println("bye!")
  }

}
