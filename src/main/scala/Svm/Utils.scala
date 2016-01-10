package Svm

import breeze.linalg._
import breeze.numerics.sqrt
import breeze.plot._
import breeze.stats.distributions.MultivariateGaussian

import scala.util.Random
import breeze.plot._
import breeze.numerics.exp
/**
  * Created by nikolaas steenbergen on 1/8/16.
  */
object Utils {

  /**
  def GaussianKernel(X1, X2, sigma):
    assert(X1.shape[0] == X2.shape[0])
    K = cdist(X1.T, X2.T, 'euclidean')
    K = sp.exp(-(K ** 2) / (2. * sigma ** 2))
    return K
    * @param X
    */

  def gaussianKernel(x1: DenseVector[Double], x2: DenseMatrix[Double],sigma: Double): DenseVector[Double] = {

    assert(x1.length == x2.rows)
    var i = 0
    var diff: DenseVector[Double] = DenseVector.zeros[Double](x2.cols)
    while(i < x2.cols){
      diff(i) = sqrt( (x1 :- x2(::,i)).dot(x1 :- x2(::,i)) )
      i += 1
    }

    var difff = - (diff :* diff)

    var denominator = 2 * sigma * sigma
    var nominator = difff
    var expon = nominator / denominator
    var ret = exp( difff :/ (2 * sigma * sigma) )

    ret
  }


  def predict_svm_kernel_all(x: DenseVector[Double],X: DenseMatrix[Double], W: DenseVector[Double], sigma: Double): Double = {
    var ret = W.t * gaussianKernel(x,X,sigma)
    return ret
  }

  /**
    *
  def fit_svm_kernel(W,X,Y,its=100,eta=1.,C=.1,kernel=(GaussianKernel,(1.)),visualize=False):
    D,N = X.shape[0],X.shape[1]
    X = sp.vstack((sp.ones((1,N)),X))

    errors_std_loc = []
    for it in range(its):
      errors_std_loc.append(test_svm(X,Y,W,kernel)[0])
      if visualize:
        #print "discount:",discount
        plot(errors_std_loc)

      rn = sp.random.randint(N)
      yhat = predict_svm_kernel(X[:,rn],X,W,kernel)
      discount = eta/(it+1.)
      if yhat*Y[:,rn] > 1: G = C * W
      else: G = C * W - Y[:,rn] * kernel[0](sp.vstack((X[:,rn] )),X,kernel[1]).flatten()

      W -= discount * G
    return W,errors_std_loc

    * @param X
    */

  def fit_svm_kernel(W: DenseVector[Double], X_org: DenseMatrix[Double], Y: DenseMatrix[Double], iterations: Int, eta: Double = 1.0, C: Double = 0.1, sigma:Double = 1.0): (DenseVector[Double],DenseVector[Double]) = {
    var D: Int = X_org.rows
    var N: Int = X_org.cols
    var X = DenseMatrix.vertcat[Double](DenseMatrix.ones[Double](1,N),X_org)


    var errors:DenseVector[Double] = DenseVector.zeros[Double](iterations);
    for(i <- 0 until iterations){
      var err = test_svm(X,Y,W,sigma)
      errors(i) = err
      println("iteration:" + i + "erorr:" + errors(i))
      //var rn = Random.nextInt(N)
      var rn = i%N
      var yhat: Double = predict_svm_kernel_all(X(::,rn),X,W,sigma)
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

  /**
    * def test_svm(X,Y,W,(k,(kparam))):
        kernel = (k,(kparam))
        error = np.zeros(1)
        point_error = 0
        for rn in range(X.shape[1]):
          yhat = predict_svm_kernel(X[:,rn],X,W,kernel)
          err = yhat*Y[:,rn]
          if not err >= 0:
            error -= yhat*Y[:,rn]
            point_error += 1
        return [error[0]/float(X.shape[1]),point_error/float(X.shape[1])]
    * @param X
    */
  def test_svm(X: DenseMatrix[Double], Y: DenseMatrix[Double],W: DenseVector[Double],sigma: Double): Double = {
    var i: Int = 0
    var N: Int = X.cols

    var point_error: Int = 0
    var error: Double = 0.0d

    while( i < N){
      var yhat = predict_svm_kernel_all(X(::,i),X,W,sigma)
      var err:DenseVector[Double] = yhat * Y(::,i)

      // TODO: change
      assert(err.length == 1)

      if(!(err(0) >= 0) ){
        error -= err(0)
        point_error += 1
      }
      i += 1
    }
    return error/N.toDouble
  }

  def plotLine(X: DenseVector[Double]): Unit = {
    val f = Figure()
    val p = f.subplot(0)

    val x_ints = DenseVector.range(0,X.length,1)
    val x = convert(x_ints, Double)
    p += plot(x,X,'-')
    p.xlabel = "x axis"
    p.ylabel = "y axis"
  }

  def plotData(X: DenseMatrix[Double]): Unit= {
    assert(X.rows == 2)
    val f = Figure()
    val p = f.subplot(0)
    //val x = linspace(0.0,1.0)
    p += plot(X(0,::).t,X(1,::).t,'.')
    //    p += plot(x, x :^ 3.0, '.')
    p.xlabel = "x axis"
    p.ylabel = "y axis"
  }

  def plotModel(X: DenseMatrix[Double], W: DenseVector[Double], sigma: Double): Unit ={
    // classify all points:
    var class0: DenseMatrix[Double] = null
    var class1: DenseMatrix[Double] = null
    for( i <- 0 until X.cols){
      // compute prediction
      var res = predict_svm_kernel_all(X(::,i),X,W,sigma)

      if (res < 1){

        if(class0 != null){
          class0 = DenseMatrix.horzcat(class0, X(::,i).toDenseMatrix.t )
        }else{
          class0 = X(::,i).toDenseMatrix.t
        }
      } else {
        if ( class1 != null) {
          class1 = DenseMatrix.horzcat(class1, X(::,i).toDenseMatrix.t )
        } else {
          class1 = X(::,i).toDenseMatrix.t
        }
      }
    }

    // put in plots
    val f = Figure()
    val p = f.subplot(0)
    if(class0 != null){
      println("class0:",class0.cols,class0.rows)
      println(class0)
      p += plot(class0(0,::).t,class0(1,::).t,'.')
    }
    if(class1 != null){
      println("class0:",class1.cols,class1.rows)
      println(class1)
      p += plot(class1(0,::).t,class1(1,::).t,'.')
    }
  }

  def make_data_xor(N: Int = 80, noise: Double = 0.25d): (DenseMatrix[Double],DenseMatrix[Double]) = {

    assert(N%4 == 0,"please use multiple of 4 for number of random samples")
    var mu: DenseMatrix[Double] = DenseMatrix((-1.0,1.0), (1.0,1.0)).t
    var C = DenseMatrix.eye[Double](2) * noise;
    println("C\n" + C)

    // sample and convert to matrix
    var mvn1 = MultivariateGaussian(mu(::,0),C)
    var samples1 = mvn1.sample(N/4)

    var mvn2 = MultivariateGaussian(-mu(::,0),C)
    var samples2 = mvn2.sample(N/4)

    var mvn3 = MultivariateGaussian(mu(::,1),C)
    var samples3 = mvn3.sample(N/4)

    var mvn4 = MultivariateGaussian(-mu(::,1),C)
    var samples4 = mvn4.sample(N/4)

    // concat matrices TODO: this probably can be done smarter
    //X = sp.hstack((mvn(mu[:,0],C,N/4).T,mvn(-mu[:,0],C,N/4).T, mvn(mu[:,1],C,N/4).T,mvn(-mu[:,1],C,N/4).T))
    var totalSamplesLength = samples1.length + samples2.length + samples3.length + samples4.length
    var X = DenseMatrix.zeros[Double](2,totalSamplesLength)
    var numSamplesProcessed: Int = 0
    for(i <- 0  until samples1.length){
      X(::,i) := samples1(i)
    }
    numSamplesProcessed += samples1.length

    for(i <- 0  until samples2.length){
      X(::,i + numSamplesProcessed) := samples2(i)
    }
    numSamplesProcessed += samples2.length

    for(i <- 0  until samples3.length){
      X(::,i + numSamplesProcessed) := samples3(i)
    }

    numSamplesProcessed += samples3.length
    for(i <- 0  until samples4.length){
      X(::,i + numSamplesProcessed) := samples4(i)
    }

    // Y = sp.hstack((sp.ones((1,N/2.)),-sp.ones((1,N/2.))))
    var ones: DenseMatrix[Double] = DenseMatrix.ones[Double](1,N/2);
    var Y: DenseMatrix[Double] = DenseMatrix.horzcat(ones.copy,-ones.copy)

    return (X,Y)
  }
}

