package me.lorand.slogreg.optimize

import org.apache.spark.rdd.RDD
import me.lorand.slogreg.helpers._

trait Optimizer {
  def iterate(data: RDD[LabeledExample]): Unit
}
