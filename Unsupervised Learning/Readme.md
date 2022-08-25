

# Document Clustering using Jaccard Index

## Author
- Rishika Tibrewal
- Shreyansh Rastogi

## Overview

In each of the text collections **KOS blog entries**, **NIPS full
papers**, and **Enron Emails**, each document is summarized as a bag of
words. The individual documents are identified by document IDs and the
words are identified by word IDs. After some cleaning up, in each
collection the vocabulary of unique words has been truncated to only
keep words that occurred more than ten times overall in that
collection.\
\
[**Click here**](https://archive.ics.uci.edu/ml/datasets/Bag+of+Words)
to go to the UCI repository.

## Objective

The task is to cluster the documents in these datasets via **K-means
clustering** for different values of K and determine an optimum value of
K.

## Methodology

We used **Jaccard Index** as a similarity measure between two documents
based on the overlap of words present in both documents.
$$J(D_1,D_2)= \frac{| D_1\cap D_2|}{|D_1\cup D_2|}$$\
For K-Means Clustering, **Jaccard distance** is used as a measure of
distance between a point and a centroid. $$D(D_1,D_2)= 1-J(D_1,D_2)$$
The value of k is optimised using the **elbow method** where we minimise
the **inertia** (the sum of squared distances of samples to their
closest cluster centroid).\
\
The libraries used are **NumPy**, **Pandas**, **Matplotlib**, **time**,
**memory_profiler**, and **random**.

## Implementation

-   We uploaded the .txt files (skipping the first 3 lines) to create a
    dataframe with column names DocID, WordID, and Count.

-   The **Enron Emails** dataset was large and due to lack of
    computational power, we decided to work on a subset of the dataset.
    For the purpose, we picked out the documents with more than 200
    distinct words, and worked on it.

-   We created the sparse Matrix where each column represents the
    presence or absence of a particular word in each of the documents,
    ignoring the multiplicity of the word (assigning 1 if the word is
    present in the document, 0 otherwise)

-   Using the Sparse Matrix we calculated the Jaccard distance between
    any two documents.

-   We applied K-Means Clustering from scratch using Jaccard Distance.
    We initialised k centroids (one for each cluster) randomly. The
    following steps are performed iteratively for some maximum number of
    assigned iterations.

    -   We assign the documents to the cluster where the Jaccard
        distance between the centroid and that document is minimum.

    -   We reassigned the centroids with the document having the lowest
        sum of Jaccard distances in each cluster.

-   We used the elbow method to optimise the number of clusters using
    the value of k with the minimum inertia.

-   Output is in the form of a dictionary where the keys are the cluster
    labels and the values are the DocID belonging to that cluster.

-   Using PCA, we tried to visualise the clusters in 3D.

## Results

We plotted the graph of inertia vs number of cluster points to find a
sharp elbow and hence get the optimum value of k.

### KOS blog entries
![Elbow kos](https://user-images.githubusercontent.com/94676910/185742806-2c42b7c8-847b-4063-9e43-2d38800cc2d5.png)

The optimum value of k is 3.

### NIPS full papers

![elbow nips](https://user-images.githubusercontent.com/94676910/185742817-c03c70fd-fb35-46f0-81aa-1cf62a775e7f.png)


The optimum value of k is 3.

### Enron Emails

![elbow enron](https://user-images.githubusercontent.com/94676910/185742820-b1b6c243-82e6-42cc-a742-3fadb4dee52f.png)


The optimum value of k is 3.

## Comparison of Measures

 |              **Measure**            | **KOS** | **NIPS** |**Enron**
 |-------------------------------------|---------|----------|-----------
 |         No. of Clusters (K)         |    3    |    3     |     3
 |        Time for K-Means (s)         |   30    |    11    |    37
 | Time for Jaccard distance matrix (s)|   59    |    20    |    283
 |             Space (MiB)             |  1177   |   1550   |   2823

