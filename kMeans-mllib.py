from __future__ import print_function
from pyspark import SparkContext
from pyspark.mllib.feature import HashingTF
from pyspark.mllib.feature import IDF
from itertools import islice
from pyspark.mllib.clustering import KMeans, KMeansModel
from numpy import array
from math import sqrt
import numpy as np
import sys

if __name__ == "__main__":
    """
        k-means
    """

    def doc_to_words(val_index, multiplier):
        temp = ""
        for i, val in enumerate(collectVocab):
            if((val_index - 1) == i):
                for x in xrange((multiplier)):
                    temp  = temp + " " + val
                final = temp[1:]
                return final
        

    sc = SparkContext(appName="kMeans")

    documents = sc.textFile("docword.pubmed.txt").map(lambda line: line.encode("utf8").split(" "))
    vocab = sc.textFile("vocab.pubmed.txt")

    counters = documents.take(3)
    
    num_docs = int(counters[0][0])
    vocab_size = int(counters[1][0])
    sum_words = int(counters[2][0])

    # collectVocab = vocab.collect()
    # remove top 3 lines from document
    doc_wo_counters = documents.mapPartitionsWithIndex(lambda i, iter: islice(iter, 3, None) if i == 0 else iter)

    final_doc = doc_wo_counters.map(lambda x: (int(x[0]), doc_to_words(int(x[1]), int(x[2])).encode("utf8"))).reduceByKey(lambda x, y: x + " " + y)

    vect_rep = final_doc.map(lambda x: x[1])

    raw_document = sc.textFile("test.txt")
    vect_rep = raw_document.map(lambda line: line.encode("utf8").split(" "))

    
    # TfIDF
    hashingTF = HashingTF()
    tf = hashingTF.transform(vect_rep)
    tf.cache()
    idf = IDF().fit(tf)
    tfidf_vectors = idf.transform(tf)
    
    #Build the model (cluster the data)
    clusters = KMeans.train(tfidf_vectors, 10, maxIterations=100)
    
    # Evaluate clustering by computing Within Set Sum of Squared Errors
    def error(point):
        center = clusters.centers[clusters.predict(point)]
        return sqrt(sum([x**2 for x in (point.toArray() - center)]))

    WSSSE = tfidf_vectors.map(lambda point: error(point)).reduce(lambda x, y: x + y)
    print("Within Set Sum of Squared Error = " + str(WSSSE))

    # Save and load model
    clusters.save(sc, "myModelPath")
    sameModel = KMeansModel.load(sc, "myModelPath")
    sc.stop()