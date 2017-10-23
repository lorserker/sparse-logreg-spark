# Lessons Learned while Implementing a Sparse Logistic Regression Algorithm in Apache Spark

This repository contains the code examples for the [Spark Summit EU 2017 talk](https://spark-summit.org/eu-2017/events/lessons-learned-while-implementing-a-sparse-logistic-regression-algorithm-in-apache-spark/)

For the experiments this [dataset](https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary.html#avazu) was used. (the `avazu-site.tr.bz2` file)

Run any of the examples like this:

```
sbt runMain me.lorand.slogreg.jobs.MyExampleJob <path_to_data> <number_of_iterations>
```

### Sparse matrix gradient descent

The optimization is implemented in `me.lorand.slogreg.optimize.SparseMatrixGradientDescent`

The job without a known partitioner is `me.lorand.slogreg.jobs.SparseMatrixJob` and the job with a known partitioner is 
`me.lorand.slogreg.jobs.SparseMatrixPartitionerJob`

### Gradient descent without joins

The version without joins is implemented in `me.lorand.slogreg.optimize.GradientDescent` and the corresponding experiment
is run by the job `me.lorand.slogreg.jobs.GradientDescentJob`

### Gradient descent with `aggregate`

Implemented in `me.lorand.slogreg.optimize.AggregateGradientDescent`

### Mini batch gradient descent

Also uses `AggregateGradientDescent` and the experiment runs in the job `me.lorand.slogreg.jobs.MiniBatchJob`

### ADAM

`me.lorand.slogreg.optimize.Adam` extends `AggregateGradientDescent`, and the experiment is run in
`me.lorand.slogreg.jobs.MomentumJob`

##### Time per iteration

Measured on an AWS EMR cluster of 5 m4.2xlarge nodes

![time per iteration](https://github.com/lorserker/sparse-logreg-spark/blob/master/img/time_per_iteration.png)

The initial version is almost 4 minutes, the best version is half a second.

