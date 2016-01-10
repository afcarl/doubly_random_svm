package Svm
import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by nikolaas steenbergen on 1/8/16.
  */

object Tests {


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

  def main(args:Array[String]):Unit = {
    println("gaussian kernel")
    runSample()
  }

}
