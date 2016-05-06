package Svm.dsekl

import java.util.concurrent.Callable

import breeze.linalg.{Axis, DenseMatrix, DenseVector, sum}
import breeze.numerics.{exp, signum}
import breeze.stats._
/**
  * Created by nikste on 5/4/16.
  */



class DSEKLBATCH(
  n_expand_samples: Int = 100,
  n_pred_samples: Int = 100,
  n_its: Int = 100,
  eta: Double=1.0,
  C: Double = 0.001,
  gamma: Double = 0.1,
  workers: Int = 1,
  damp: Boolean = true,
  validation: Boolean = false,
  verbose: Boolean = false) {

  var X_pred_ids = List[DenseVector[Int]]()
  var X_exp_ids = List[DenseVector[Int]]()

  // TODO: bad scala style, change if there's time
  var w: DenseVector[Double] = null
  var X: DenseMatrix[Double] = null
  var y: DenseVector[Double] = null

  def createBatchIndices(Xinlength: Int, batchsize: Int): List[DenseVector[Int]] = {
    var num_batches = Xinlength / batchsize.toDouble
    var batchIds = List[DenseVector[Int]]()
    // normal
    for( i <- 0 until num_batches.toInt){
      batchIds :+ DenseVector.range(batchsize * i, (i + 1) * batchsize)
    }
    // rest
    if((num_batches - num_batches.toInt) > 0){
      //TODO: shouldn't this be Xinlength - 1 ?
      batchIds :+ DenseVector.range(batchsize * num_batches.toInt, Xinlength)
    }
    return batchIds
  }

    def fit(Xin: DenseMatrix[Double], yin: DenseVector[Double]): Unit ={
      var traintestsplit: Int = 0
      var Xval: DenseMatrix[Double] = null;
      var Yval: DenseVector[Double] = null;


      //TODO: insert random permutation of training data
      if(this.validation){
        traintestsplit =  (yin.length * 0.002).toInt;

        Xval = Xin(0 to traintestsplit,::)
        Yval = yin(0 to traintestsplit)
        X = Xin(traintestsplit to -1,::)
        y = yin(traintestsplit to -1)
      }else{
        X = Xin
        y = yin
      }

      var num_batch_pred = X.rows / n_pred_samples.toDouble
      var num_batch_exp = X.rows / n_expand_samples.toDouble

      X_pred_ids = createBatchIndices(X.rows,n_pred_samples)
      X_exp_ids = createBatchIndices(X.rows,n_expand_samples)

      if(this.verbose){
        if(this.validation){
          println ("Training DSEKL on " + y.size + " and validation on " + traintestsplit)
        }else{
          println ("Training DSEKL on " + y.size)
        }
        println("Hyperparameters:\n")
        println("n_expand_samples:" + this.n_expand_samples)
        println("n_pred_samples:" + this.n_pred_samples)
        println("eta_start:" + this.eta)
        println("C:" + this.C)
        println("gamma:" + this.gamma)
        println("workers:" + this.workers)
        println("damp:" + this.damp)
        println("validation:" + this.validation)
      }

      w = DenseVector.rand[Double](y.size)
      var G = DenseVector.ones[Double](y.size)
      // TODO: insert tracking of validation errors here
      //      if self.validation:
      //        self.valErrors = []
      // TODO: insert tracking of training errors here
      // self.trainErrors = []

      var it: Int = 0
      var delta_w: Double = 50.0
      while(it < this.n_its && delta_w > 1.0){
        if(this.verbose && it != 1){
          println("iteration" + it + " of " + this.n_its)
          if(this.validation){
            var pred: DenseVector[Double] = this.predict(Xval)
            var e: DenseVector[Double] = signum(pred)
//            var val_error: Double = mean( != Yval)
//            val_error = (sp.sign(self.predict(Xval)) != Yval).mean()
          }
        }
      }

      print("fitting!")
    }

  def outer(x1: DenseMatrix[Double], x2: DenseVector[Double]): DenseMatrix[Double] = {
    assert(x1.cols == 1)
    x1 * x2.t
  }
  def outer(x1: DenseMatrix[Double], x2: DenseMatrix[Double]): DenseMatrix[Double] = {
    assert(x1.cols == 1)
    assert(x2.cols == 1)
    x1 * x2.t
  }
  def outer(x1: DenseVector[Double], x2: DenseVector[Double]): DenseMatrix[Double] = {
    x1 * x2.t
  }

  def GaussKernMini(X1: DenseMatrix[Double], X2: DenseMatrix[Double], sigma: Double): DenseMatrix[Double] = {
    //G = sp.outer((X1 * X1).sum(axis=0),sp.ones(X2.shape[1]))
    var G: DenseMatrix[Double] = outer(sum(X1 :* X1,Axis._0),DenseVector.ones[Double](X2.cols))
    //H = sp.outer((X2 * X2).sum(axis=0),sp.ones(X1.shape[1]))
    var H: DenseMatrix[Double] = outer(sum(X2 :* X2,Axis._0),DenseVector.ones[Double](X1.cols))
    //K = sp.exp(-(G + H.T - 2.*(X1.T.dot(X2)))/(2.*sigma**2))
    var one: DenseMatrix[Double] = X1.t
    var in = one * X2
    var K = exp(-(G + H.t - 2.0 * in)/(2.0 * sigma * sigma))

    return K
  }

  def svm_predict_raw_batches(Xtrain: DenseMatrix[Double], Xtest: DenseMatrix[Double], w: DenseVector[Double], ids_expand: DenseVector[Int], sigma: Double = 1.0): DenseVector[Double] ={
    var xtrain: DenseMatrix[Double] = Xtrain(ids_expand,::)
    var K = GaussKernMini(Xtest.t, xtrain.t, sigma)
    return K.dot(w(ids_expand))
  }


  def predict(Xtest: DenseMatrix[Double]): DenseVector[Double] ={
//    var num_batches = Xtest.rows / this.n_pred_samples.toDouble
    val test_ids = createBatchIndices(Xtest.rows,this.n_pred_samples)

    var yhattotal = List[DenseVector[Double]]()
    for (i <- 0 until test_ids.size){
//      yraw = Parallel(n_jobs=self.workers)(delayed(svm_predict_raw_batches)(self.X, Xtest[test_ids[i]], \
//        self.w, v, self.gamma) for v in self.X_exp_ids)
      for (j <- 0 until this.X_exp_ids.size){

        val predictionStep = new Thread(new ComputePrediction(this.X,Xtest(test_ids(i)),this.w,this.X_exp_ids(j),this.gamma))
      }
//      yhat = sp.sign(sp.vstack(yraw).mean(axis=0))
//      yhattotal.append(yhat)

    }
  }
}
class ComputePrediction(val X: DenseMatrix[Double],
              val Xtest: DenseMatrix[Double],
              val test_ids: DenseVector[Int],
              val w: DenseVector[Double],
              val v: DenseVector[Int],
              val gamma: Double) extends Callable {

  override def call(): DenseVector[Double] = {
    svm_predict_raw_batches(X,Xtest,w,test_ids,gamma)
  }

  def svm_predict_raw_batches(Xtrain: DenseMatrix[Double], Xtest: DenseMatrix[Double], w: DenseVector[Double], ids_expand: DenseVector[Int], sigma: Double = 1.0): DenseVector[Double] ={
    var xtrain: DenseMatrix[Double] = Xtrain(ids_expand,::)
    var K = GaussKernMini(Xtest.t, xtrain.t, sigma)
    return K.dot(w(ids_expand))
  }

  def GaussKernMini(X1: DenseMatrix[Double], X2: DenseMatrix[Double], sigma: Double): DenseMatrix[Double] = {
    //G = sp.outer((X1 * X1).sum(axis=0),sp.ones(X2.shape[1]))
    var G = outer(sum(X1 :* X1,Axis._0),DenseVector.ones[Double](X2.cols))
    //H = sp.outer((X2 * X2).sum(axis=0),sp.ones(X1.shape[1]))
    var H = outer(sum(X2 :* X2,Axis._0),DenseVector.ones[Double](X1.cols))
    //K = sp.exp(-(G + H.T - 2.*(X1.T.dot(X2)))/(2.*sigma**2))
    var K = exp(-(G + H.t - 2.0 * (X1.t.dot(X2)))/(2.0 * sigma * sigma))
    return K
  }

  def outer(x1: DenseMatrix[Double], x2: DenseVector[Double]): DenseMatrix[Double] = {
    assert(x1.cols == 1)
    x1 * x2.t
  }
}