package me.lorand.slogreg.optimize
import me.lorand.slogreg.helpers._
import org.apache.spark.rdd.RDD

class AggregateGradientDescent(nRows: Long, batchSize: Int, nDim: Int, learningRate: Double, lambda: Double, initBias: Option[Double] = None) extends Serializable {

  val weights: Array[Double] = Array.ofDim[Double](nDim + 1)
  var b: Double = initBias.getOrElse(0d)

  var iteration = 0

  def iterate(data: RDD[LabeledExample]): Unit = {
    iteration += 1
    val (gradient, db) = getGradient(data)

    // update weights
    var k = 0
    while (k < weights.length) {
      weights(k) = weights(k) - learningRate * gradient(k)
      k += 1
    }
    b = b - learningRate * db
  }

  def getGradient(data: RDD[LabeledExample]): (Array[Double], Double) = {
    val aggregator = new GradientAggregator(nDim, weights, b, lambda)
    val seqOp = (agg: GradientAggregator, example: LabeledExample) => agg.seqOp(example)
    val combOp = (agg1: GradientAggregator, agg2: GradientAggregator) => agg1.combOp(agg2)

    val (batch, size) = sampleData(data, batchSize)

    val result = batch.aggregate(aggregator)(seqOp, combOp)

    (result.gradient.map(_ / size), result.db / size)
  }

  def sampleData(data: RDD[LabeledExample], n: Int): (RDD[LabeledExample], Long) = {
    if (n == nRows) {
      (data, nRows)
    } else {
      val fraction = n.toDouble / nRows
      val result = data.sample(true, fraction)
      (result, batchSize)
    }
  }
}

class GradientAggregator(nDim: Int, weights: Array[Double], b: Double, lambda: Double) extends Serializable {
  val gradient: Array[Double] = Array.fill(nDim + 1)(0d)
  var db: Double = 0.0

  def seqOp(example: LabeledExample): this.type = {
    val (target, indexes, values) = (example.target, example.indexes, example.values)

    var dotProd = b
    var k = 0
    while (k < indexes.length) {
      dotProd += values(k) * weights(indexes(k))
      k += 1
    }
    k = 0
    while (k < indexes.length) {
      gradient(indexes(k)) += (sigmoid(dotProd) - target) * values(k) + lambda * weights(indexes(k))
      k += 1
    }
    db = db + (sigmoid(dotProd) - target)
    this
  }

  def combOp(other: GradientAggregator): this.type = {
    var k = 0
    while (k < gradient.length) {
      gradient(k) = gradient(k) + other.gradient(k)
      k += 1
    }
    db = db + other.db
    this
  }
}
