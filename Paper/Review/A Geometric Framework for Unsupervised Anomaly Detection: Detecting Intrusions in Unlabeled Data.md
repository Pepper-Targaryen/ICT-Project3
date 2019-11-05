# A Geometric Framework for Unsupervised Anomaly Detection: Detecting Intrusions in Unlabeled Data

## Author

Eleazar Eskin, Andrew Arnold, Michael J. Prerau, Leonid Portnoy, Salvatore J. Stolfo

## Conference

Applications of Data Mining in Computer Security 2002

## Resume

In intrusion detection, supervised learning is not very suitable because labelling records are very expensive.
Therefore, we try to find unsupervised learning methods for detection.

In this article, we first define features and second introduce 3 methods for detection.

For defining features, it gives two methods: 
* feature space
* kernel function

However, in this article there is no specification expression for feature space.

For detection, we have 3 methods:
* Clustering based methods
* KNN
* One-class SVM

By clustering based methods, we consider a sphere of each point. The points contains many other points are normal. Otherwise, it can be 
anormal.

By KNN, we consider the sum of distance of K nearest neighbors. If the sum is larger than a threshold, it may be anormal.

By SVM, we separete the majority of points from original. The rest points are anormal.
