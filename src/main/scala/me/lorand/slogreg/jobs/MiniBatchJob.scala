package me.lorand.slogreg.jobs

import me.lorand.slogreg.helpers.{LabeledExample, loadLibSVM}
import me.lorand.slogreg.optimize.AggregateGradientDescent
import org.apache.spark.{SparkConf, SparkContext}
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory
import me.lorand.slogreg.helpers._

object MiniBatchJob {
  lazy val logger = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]): Unit = {
    val datafile = args(0)
    val nIter = args(1).toInt

    val conf = new SparkConf().setAppName("Main").setMaster("local[*]")
    val sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("WARN")

    val data: RDD[LabeledExample] = loadLibSVM(sc, datafile).persist()


    val nRows: Long = data.count
    val nDim: Int = data.map(x => x.indexes.max).max

    logger.warn(s"we have $nRows rows and $nDim dimensions")

    val optimizer = new AggregateGradientDescent(nRows, 1024, nDim, learningRate = 0.1, lambda = 0.0001, Some(math.log(0.2) - math.log(0.8)))

    val iterationTimes = Array.ofDim[Long](nIter)
    val losses = Array.ofDim[Double](nIter)

    for (i <- 0 until nIter) {
      val startTime = System.currentTimeMillis()
      optimizer.iterate(data)
      iterationTimes(i) = System.currentTimeMillis() - startTime
      logger.warn(s"iter $i")
      losses(i) = logloss(optimizer.weights, optimizer.b, data, nRows)
      logger.warn(s"loss ${losses(i)}")
    }

    logger.warn(s"learningCurve = ${losses.mkString("[", ",", "]")}")
    logger.warn(s"avgIterationTime = ${iterationTimes.map(_ / nIter).sum}")

    sc.stop()
  }
}
