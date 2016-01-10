package Svm

import breeze.linalg._

/**
  * Created by nikolaas steenbergen on 1/8/16.
  */
object Main {


  def main(args : Array[String]): Unit ={
    println("herro!")

    var ret = Utils.make_data_xor(500,0.1);
    var X = ret._1
    var Y = ret._2

    println("X:\n" + X)
    println("Y: \n" + Y)

    var N = X.cols

    var W: DenseVector[Double] = DenseVector.rand(N)

    var (w,errors) = Utils.fit_svm_kernel(W, X, Y, iterations = 1000, eta = 1.0, C = 0.1)

    var sigma: Double = 1.0

    Utils.plotData(X)
    Utils.plotLine(errors)
    Utils.plotModel(X, w, sigma)
    println("bye!")
  }

}
