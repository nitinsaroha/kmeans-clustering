from __future__ import print_function
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from itertools import islice
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.linalg import SparseVector, DenseVector
import numpy as np
import sys

def doc_to_words(val_index, multiplier):
    temp = ""
    for i, val in enumerate(collectVocab):
        if((val_index - 1) == i):
            for x in xrange((multiplier)):
                temp  = temp + " " + val
            final = temp[1:]
            return final

def assign_point(line, centroid):
    index = 0
    max_distance = float("+inf")
    for k in range(len(centroid)):
        squared_dist = line.squared_distance(centroid[k])
        if squared_dist < max_distance:
            max_distance = squared_dist
            index = k
    return index

def add(v1, v2):
    """Add two sparse vectors
    >>> v1 = Vectors.sparse(3, {0: 1.0, 2: 1.0})
    >>> v2 = Vectors.sparse(3, {1: 1.0})
    >>> add(v1, v2)
    SparseVector(3, {0: 1.0, 1: 1.0, 2: 1.0})
    """
    assert isinstance(v1, SparseVector) and isinstance(v2, SparseVector)
    assert v1.size == v2.size
    indices = set(v1.indices).union(set(v2.indices))
    v1d = dict(zip(v1.indices, v1.values))
    v2d = dict(zip(v2.indices, v2.values))
    zero = np.float64(0)
    values =  {i: v1d.get(i, zero) + v2d.get(i, zero)
       for i in indices
       if v1d.get(i, zero) + v2d.get(i, zero) != zero}

    return Vectors.sparse(v1.size, values)

def Average(clusters):
    clusters_len = len(clusters)
    out = clusters[0]
    if clusters_len > 1:
        for x in xrange(clusters_len):
            out = add(out, clusters[x])
        avg_vector = SparseVector(out.size, out.indices, out.values / clusters_len)
    else:
        avg_vector = out
    return avg_vector

if __name__ == "__main__":
    """
        k-means
    """

    sc = SparkContext(appName="kMeans")

    documents = sc.textFile("docword.nips.txt").map(lambda line: line.encode("utf8").split(" "))
    vocab = sc.textFile("vocab.nips.txt")

    counters = documents.take(3)
    
    num_docs = int(counters[0][0])
    vocab_size = int(counters[1][0])
    sum_words = int(counters[2][0])

    collectVocab = vocab.collect()
    
    doc_wo_counters = documents.mapPartitionsWithIndex(lambda i, iter: islice(iter, 3, None) if i == 0 else iter)

    final_doc = doc_wo_counters.map(lambda x: (int(x[0]), doc_to_words(int(x[1]), int(x[2])).encode("utf8"))).reduceByKey(lambda x, y: x + " " + y)

    vect_rep = final_doc.map(lambda x: x[1])

    # raw_document = sc.textFile("test.txt")
    # raw_document.cache()
    # vect_rep = raw_document.map(lambda line: line.encode("utf8").split(" "))
    
    # TfIDF
    hashingTF = HashingTF()
    tf = hashingTF.transform(vect_rep)
    tf.cache()
    idf = IDF().fit(tf)
    tfidf_vectors = idf.transform(tf)
        
    k = 10
    new_centroids = None
    # first argument whether sample may contain multiple copies of same record
    # second is the number of samples to be taken
    # third is the seed which is a given random number generator
    old_centroids = tfidf_vectors.takeSample(False, k, 42)

    iterations = 0
    converge = 1.0
    x = 100
    t = 0.01

    while iterations < x and converge > t:
        points = tfidf_vectors.map(lambda line: (assign_point(line, old_centroids), line))    
        points_combined = points.groupByKey().map(lambda x: (x[0], list(x[1])))
        new_centroids = points_combined.map(lambda x: (Average(x[1]))).collect()
        
        if(len(old_centroids) != len(new_centroids)):
            for i in range(len(old_centroids) - 1):
                converge += new_centroids[i].squared_distance(old_centroids[i])
        else:
            for i in range(len(old_centroids)):
                converge += new_centroids[i].squared_distance(old_centroids[i])

        old_centroids = new_centroids
        iterations += 1
    
    # final.coalesce(1, True).saveAsTextFile("output")
    print("Took %s iterations to converge" %  iterations)
    
    print("%s Central residual Left" %  converge)
    
    for p in old_centroids:
        print ("%s: " % p)

    sc.stop()