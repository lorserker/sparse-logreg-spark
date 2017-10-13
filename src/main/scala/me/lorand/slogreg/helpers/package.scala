package me.lorand.slogreg

import org.apache.spark.SparkContext
import org.apache.spark.rdd.RDD

package object helpers {

  case class LabeledExample(target: Double, indexes: Array[Int], values: Array[Double])

  def loadLibSVM(sc: SparkContext, filepath: String): RDD[LabeledExample] = {
    val lines = sc.textFile(filepath)
    lines.map(_.split(" "))
      .map(cols => {
        val label = cols.head.toDouble
        val (indexes, values) = cols.tail.map(col => {
          val i = col.indexOf(":")
          (col.slice(0, i).toInt, col.slice(i+1, col.length).toDouble)
        }).unzip
        LabeledExample(label, indexes, values)
      })
  }

  def loadSparseMatrixDataLibSVM(sc: SparkContext, filepath: String): (RDD[(Long, Double)], RDD[(Long, Seq[(Int, Double)])]) = {
    val lines = sc.textFile(filepath)
    val colLines: RDD[(Array[String], Long)] = lines.map(_.split(" ")).zipWithIndex
    val y: RDD[(Long, Double)] = colLines.map{ case (cols, i) => (i, cols.head.toDouble) }
    val X: RDD[(Long, Seq[(Int, Double)])] =
      colLines.map{ case (cols, i) => (i, cols.tail.map(_.split(":")).map(parts => (parts(0).toInt, parts(1).toDouble)).toSeq) }

    (y, X)
  }

  def logloss(label: Double, z: Double): Double = {
    if (label == 0) {
      math.log1p(math.exp(-z)) + z
    } else {
      math.log1p(math.exp(-z))
    }
  }

  def logloss(weights: Array[Double], b: Double,  data: RDD[LabeledExample], n: Long): Double = {
    data.mapPartitions(it => {
      it.map(x => {
        val z = b + (x.indexes zip x.values).map{ case (i, v) => weights(i) * v}.sum
        logloss(x.target, z) / n
      })
    }).sum()
  }

  def logloss(weights: Array[Double], b: Double, y: RDD[(Long, Double)], matrix: RDD[(Long, Seq[(Int, Double)])], n: Long): Double = {
    val dotProds: RDD[(Long, Double)] =
      matrix.mapValues(jvals => b + jvals.map{ case (j, x) => x * weights(j) }.sum)
    dotProds.join(y).map{ case (i, (z, target)) => logloss(target, z) / n }.sum
  }

  def sigmoid(z: Double): Double = 1.0 / (1.0 + math.exp(-z))
}
