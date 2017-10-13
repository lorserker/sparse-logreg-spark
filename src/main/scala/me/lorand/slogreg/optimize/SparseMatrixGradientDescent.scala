package me.lorand.slogreg.optimize

import org.apache.spark.rdd.RDD
import me.lorand.slogreg.helpers._

class SparseMatrixGradientDescent(nRows: Long, nDim: Int, learningRate: Double, lambda: Double) extends Serializable {

  val weights: Array[Double] = Array.ofDim[Double](nDim + 1)
  var b: Double = 0d

  def iterate(y: RDD[(Long, Double)], matrix: RDD[(Long, Seq[(Int, Double)])]): Unit = {
    val dotProds: RDD[(Long, Double)] =
      matrix.mapValues(jvals => b + jvals.map{ case (j, x) => x * weights(j) }.sum)

    val predictions: RDD[(Long, Double)] =
      dotProds.mapValues(z => sigmoid(z))

    val deltas: RDD[(Long, Double)] =
      predictions.join(y).mapValues{ case (predicted, correct) => (predicted - correct)/nRows }

    val updates: Map[Int, Double] =
      matrix.join(deltas)
        .flatMap{ case (i, (jvals, d)) => jvals.map{ case (j, x) => (j, x * d) } }
        .reduceByKey(_ + _)
        .collect
        .toMap

    val db = deltas.map(_._2).sum()
    b = b - learningRate * db

    var k = 0
    while (k < nDim) {
      val gradient = updates.getOrElse[Double](k, 0) + lambda*weights(k)/nRows
      weights(k) = weights(k) - learningRate * gradient
      k += 1
    }
  }
}
