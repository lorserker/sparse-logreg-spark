package me.lorand.slogreg.optimize

import me.lorand.slogreg.helpers
import org.apache.spark.rdd.RDD

class Adam(nRows: Long, batchSize: Int, nDim: Int, learningRate: Double, lambda: Double, beta1: Double, beta2: Double, epsilon: Double = 1e-8)
  extends AggregateGradientDescent(nRows, batchSize, nDim, learningRate, lambda, Some(math.log(0.2) - math.log(0.8))) with Serializable {

  val avgGradient: Array[Double] = Array.ofDim[Double](nDim + 1)
  var avgDb: Double = 0d
  val avgSqrGradient: Array[Double] = Array.ofDim[Double](nDim + 1)
  var avgSqrDb: Double = 0d

  var t = 0

  override def getGradient(data: RDD[helpers.LabeledExample]): (Array[Double], Double) = {
    val (gradient, db) = super.getGradient(data)
    t += 1

    val newGradient = Array.ofDim[Double](gradient.length)

    var k = 0
    while (k < gradient.length) {
      avgGradient(k) = avgGradient(k) * beta1 + gradient(k) * (1 - beta1)
      avgSqrGradient(k) = avgSqrGradient(k) * beta2 + gradient(k) * gradient(k) * (1 - beta2)

      val vCorr = avgGradient(k) / (1 - math.pow(beta1, t))
      val sCorr = avgSqrGradient(k) / (1 - math.pow(beta2, t))

      newGradient(k) = vCorr / math.sqrt(sCorr + epsilon)

      k += 1
    }
    avgDb = avgDb * beta1 + db * (1 - beta1)
    avgSqrDb = avgSqrDb * beta2 + db * db * (1 - beta2)

    val newDb = (avgDb / (1 - math.pow(beta1, t))) / math.sqrt(avgSqrDb / (1 - math.pow(beta2, t)) + epsilon)

    (newGradient, newDb)
  }

}
