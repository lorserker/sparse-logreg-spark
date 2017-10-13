package me.lorand.slogreg.jobs

import org.apache.spark.rdd.RDD
import me.lorand.slogreg.helpers._
import me.lorand.slogreg.optimize._
import org.apache.spark.{SparkConf, SparkContext}
import org.slf4j.LoggerFactory


object GradientDescentJob {

  lazy val logger = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]): Unit = {
    val datafile = args(0)
    val nIter = args(1).toInt

    val conf = new SparkConf().setAppName("Main").setMaster("local[*]")
    val sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("WARN")

    val data: RDD[LabeledExample] = loadLibSVM(sc, datafile).cache()

    val nRows: Long = data.count
    val nDim: Int = data.map(x => x.indexes.max).max

    logger.warn(s"we have $nRows rows and $nDim dimensions")

    val optimizer = new GradientDescent(nRows, nDim, learningRate = 0.1, lambda = 0.0001)

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
