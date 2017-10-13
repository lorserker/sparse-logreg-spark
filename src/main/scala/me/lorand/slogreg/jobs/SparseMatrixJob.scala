package me.lorand.slogreg.jobs

import org.apache.spark.{SparkConf, SparkContext}
import me.lorand.slogreg.helpers._
import me.lorand.slogreg.optimize.SparseMatrixGradientDescent
import org.slf4j.LoggerFactory

object SparseMatrixJob {

  lazy val logger = LoggerFactory.getLogger(this.getClass)

  def main(args: Array[String]): Unit = {
    val datafile = args(0)
    val nIter = args(1).toInt

    val conf = new SparkConf().setAppName("Main").setMaster("local[*]")
    val sc = SparkContext.getOrCreate(conf)
    sc.setLogLevel("WARN")

    val (yLoaded, matrixLoaded) = loadSparseMatrixDataLibSVM(sc, datafile)
    val (y, matrix) = (yLoaded.persist(), matrixLoaded.persist())

    val nRows: Long = y.map(_._1).max + 1
    val nDim: Int = matrix.map{ case (_, kvals) => kvals.map(_._1).max }.max

    val optimizer = new SparseMatrixGradientDescent(nRows, nDim, 0.1, 0.0001)

    val iterationTimes = Array.ofDim[Long](nIter)
    val losses = Array.ofDim[Double](nIter)

    for (i <- 0 until nIter) {
      val startTime = System.currentTimeMillis()
      optimizer.iterate(y, matrix)
      iterationTimes(i) = System.currentTimeMillis() - startTime
      logger.warn(s"iter $i")
      losses(i) = logloss(optimizer.weights, optimizer.b, y, matrix, nRows)
      logger.warn(s"loss ${losses(i)}")
    }

    logger.warn(s"learningCurve = ${losses.mkString("[", ",", "]")}")
    logger.warn(s"avgIterationTime = ${iterationTimes.map(_ / nIter).sum}")

    sc.stop()
  }



}
