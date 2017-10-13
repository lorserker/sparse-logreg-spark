package me.lorand.slogreg.optimize

import me.lorand.slogreg.helpers
import org.apache.spark.rdd.RDD

class Momentum(nRows: Long, batchSize: Int, nDim: Int, learningRate: Double, lambda: Double, beta: Double)
  extends AggregateGradientDescent(nRows, batchSize, nDim, learningRate, lambda) with Serializable {

  val avgGradient: Array[Double] = Array.ofDim[Double](nDim + 1)
  var avgDb: Double = 0d

  override def getGradient(data: RDD[helpers.LabeledExample]): (Array[Double], Double) = {
    val (gradient, db) = super.getGradient(data)
    var k = 0
    while (k < gradient.length) {
      avgGradient(k) = avgGradient(k) * beta + gradient(k)
      k += 1
    }
    avgDb = avgDb * beta + db
    (avgGradient, avgDb)
  }
}
