//package Svm
//
//
///**
//  * Created by Nikolaas Steenbergen on 1/8/16.
//  */
//
//import org.sameersingh.scalaplot.Implicits._
//import org.sameersingh.scalaplot.XYPlotStyle
//
//object Plotter {
//
//  def plotLineExample(): Unit ={
//    val x = 0.0 until 2.0 * math.Pi by 0.1
//    output(GUI,xyChart( x ->(math.sin(_), math.cos(_))))
//  }
//
//
//  def n_rands(n : Int) = {
//    1 to n map{ _ => r.nextInt(100)}
//  }
//
//  def plotScatterExample(): Unit = {
//
//    val x = 0.0 until 10.0 by 0.01
//    val rnd = new scala.util.Random(0)
//
//    var randseq : Seq[(Double,Double)] = List()
//
//    for (i <- 0 until 100){
//      randseq :+ (rnd.nextDouble(),rnd.nextDouble())
//    }
//
//    output(GUI, xyChart(n_rands(100),n_rands(100),style = XYPlotStyle.Dots))
////    output(GUI, xyChart(
////       x -> Seq(Y(x, style = XYPlotStyle.Lines),
////        Y(x.map(_ + rnd.nextDouble - 0.5), style = XYPlotStyle.Dots))))
//  }
//}
