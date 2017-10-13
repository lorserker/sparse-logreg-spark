package me.lorand.slogreg.optimize

import me.lorand.slogreg.helpers._
import org.apache.spark.rdd.RDD
import org.slf4j.LoggerFactory

class GradientDescent(nRows: Long, nDim: Int, learningRate: Double, lambda: Double) extends Serializable {

  lazy val logger = LoggerFactory.getLogger(this.getClass)

  val weights: Array[Double] = Array.ofDim[Double](nDim + 1)
  var b: Double = 0d

  def iterate(data: RDD[LabeledExample]): Unit = {
    val gradient = data.flatMap(example => {
      val z = b + (example.indexes zip example.values).map{ case (i, x) => weights(i) * x}.sum
      val prediction = sigmoid(z)
      (-1, (prediction - example.target) / nRows) ::
        (example.indexes zip example.values)
          .map{ case (k, v) => (k, (prediction - example.target) * v / nRows + lambda * weights(k) / nRows)}
          .toList
    }).reduceByKey(_ + _).collectAsMap()

    val db = gradient(-1)
    b = b - learningRate * db

    // update weights
    var i = 0
    while (i < weights.length) {
      weights(i) = weights(i) - learningRate * gradient.getOrElse(i, 0.0)
      i += 1
    }
  }

}
