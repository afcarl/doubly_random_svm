package Svm

import breeze.linalg.{DenseMatrix, DenseVector}

/**
  * Created by owner on 1/10/16.
  */
class UtilsDist {

  def fit_svm_kernel(W: DenseVector[Double], X_org: DenseMatrix[Double], Y: DenseMatrix[Double], iterations: Int, eta: Double = 1.0, C: Double = 0.1, sigma:Double = 1.0): (DenseVector[Double],DenseVector[Double]) = {
    var D: Int = X_org.rows
    var N: Int = X_org.cols
    var X = DenseMatrix.vertcat[Double](DenseMatrix.ones[Double](1,N),X_org)


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
