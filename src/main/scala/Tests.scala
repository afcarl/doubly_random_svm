package Svm
import breeze.linalg.{shuffle, convert, DenseMatrix, DenseVector}

import scala.collection.mutable.ListBuffer
import scala.util.Random

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
    Utils.plotModel(X, Y, w, sigma)
  }


  def testShuffleData(): Unit = {
    println("testing shuffled data:")
    var Yordered: DenseMatrix[Double] = DenseMatrix((1d,2d,3d,4d,5d,6d,7d,8d,9d,10d))
    var Xordered: DenseMatrix[Double] = DenseMatrix((0d,1d,2d,3d,4d,5d,6d,7d,8d,9d),(0d,1d,2d,3d,4d,5d,6d,7d,8d,9d))

    var res = Utils.shuffleData(Xordered,Yordered)
    var X = res._1
    var Y = res._2

    // check result, targets are input + 1
    for(i <- 0 until X.cols){
      println(X(0,i) + 1 + " = " +  Y(0,i))
      assert(X(0,i) + 1 == Y(0,i),"for i=" + i + " X(0," + i + ") = " + X(0,i) + " != Y(" + i + ") = " + Y(0, i))
    }
  }
  def main(args:Array[String]):Unit = {
    println("herro")
    //    runSample()
//    testShuffleData()

    var res = UtilsDist.loadMnist()
    var X = res._1
    var Y = res._2
    var res2: ListBuffer[DenseMatrix[Double]] = UtilsDist.createOneVsAllTrainingsets(Y)



    println("bye")
  }

}
