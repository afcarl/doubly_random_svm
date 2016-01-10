package Svm
import breeze.linalg.DenseVector

/**
  * Created by owner on 1/8/16.
  */

object Tests {

  def testGaussianKernel(): Unit ={
    var x1 = DenseVector[Double](1.0d,1.0d)
    var x2 = DenseVector[Double](1.0d,1.1d)
    var sigma: Double = 1
    var ret = Utils.gaussianKernel(x1,x2,sigma);
    assert(ret > 0.990)

    x1 = DenseVector[Double](0.0d,0.0d);
    x2 = DenseVector[Double](10.0d,10.0d);
    ret = Utils.gaussianKernel(x1,x2,sigma)
    assert(ret < 0.1)
  }

  def testsvmpredictone(): Unit ={
    var sigma = 1
    var x1 = DenseVector[Double](1.0d,1.0d)
    var x2 = DenseVector[Double](1.0d,1.1d)
    var weight = 0.0d
    var ret = Utils.predict_svm_kernel_one(x1,x2,sigma)
//    assert(ret == 0)

    sigma = 1
    x1 = DenseVector[Double](1.0d,1.0d)
    x2 = DenseVector[Double](1.0d,1.1d)
    weight = 0.5d
    ret = Utils.predict_svm_kernel_one(x1,x2,sigma)
//    assert(ret > 0.990*0.5 && ret < 1.0)
  }

  def main(args:Array[String]):Unit = {
    println("gaussian kernel")
    testGaussianKernel()
    testsvmpredictone()
  }

}
