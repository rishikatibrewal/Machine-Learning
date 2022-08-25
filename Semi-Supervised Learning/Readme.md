# Semi-Supervised Learning on Fashion-MNIST Data

## Author
- Rishika Tibrewal
- Shreyansh Rastogi

## Overview

Fashion-MNIST is a dataset of Zalando's article images consisting of a
training set of 60,000 examples and a test set of 10,000 examples. Each
example is a 28x28 grayscale image, for a total of 784 pixels in total.
Each pixel has a single pixel-value associated with it, indicating the
lightness or darkness of that pixel, with higher numbers meaning darker.
This pixel-value is an integer between 0 and 255. Each training and test
example is assigned to one of the following labels: **T-shirt/top**,
**Trouser**, **Pullover**, **Dress**, **Coat**, **Sandal**, **Shirt**,
**Sneaker**, **Bag**, and **Ankle Boot**.

## Objective

We will be using K-Means clustering to identify a small subset of
labelled images to seed the classification process and increase the
accuracy.

## Methodology

The idea behind this project is to use K-Means Clustering for
semi-supervised learning, where we have plenty of unlabelled instances
with a very few labelled ones. For the purpose, we try to find an
optimal number of such labelled instances, **k**.\
We build two models, **Logistic Regression** & **Neural Networks** and
compare the accuracies in each case.

-   Models on the entire training dataset.

-   Models on randomly labelled k instances of the training set.

-   Models on the centroids computed using K-Means Clustering.

-   Propgating labels of the centroid to each data point in the cluster.

-   Propgating labels to data points closer to cluster centroids (using
    some percentile).

The libraries used are **NumPy**, **tensorflow**, **sklearn**,
**Matplotlib**, **time**, **memory_profiler**, and **warnings**.

## Implementation

We imported the Fashion MNIST dataset from keras. Since each datapoint
was a 28x28 grayscale image, we flattened it so that we can use it to
build our models. All the models were trained and tested on the
flattened data.

-   We first applied the standard models on the flattened train set.

     |      Model          |   Accuracy   |   Time (s)   |    Space (MiB)
     |:-------------------:|:------------:|:------------:|:---------------:
     | Logistic Regression |     84%      |     644      |      1506
     |   Neural Network    |     89%      |     191      |      1540

    We observe that the Neural Network outshines the Logistic Regression
    model when we use these models on the entire dataset.

-   We found the optimum number of labelled instances, k by taking
    the accuracy of the models into consideration (here, we fitted the
    model to the first k data points of the train set). We took k =
    100.

    |       Value of k         |   **Logistic Regression Accuracy**  |  **Neural Network Accuracy**
    |:------------------------:|:-----------------------------------:|:-------------------------------:
    |            10            |     37%                             |     36%
    |            15            |     41%                             |     39%
    |            20            |     43%                             |     41%
    |            25            |     55%                             |     48%
    |            30            |     57%                             |     49%
    |            35            |     59%                             |     60%
    |            40            |     67%                             |     59%
    |            45            |     65%                             |     62%
    |            50            |     66%                             |     62%
    |            55            |     67%                             |     67%
    |            60            |     67%                             |     68%
    |            65            |     67%                             |     67%
    |            70            |     69%                             |     70%
    |            75            |     69%                             |     68%
    |            80            |     69%                             |     72%
    |            85            |     69%                             |     71%
    |            90            |     68%                             |     70%
    |            95            |     67%                             |     72%
    |            100           |     69%                             |     72%
    |--------------------------|-------------------------------------|------------------------------
    |    Space:                |    1076 MiB                         |     1374 MiB
    |    Time:                 |    13 s                             |     63 s
    

-   We first clustered the training set into 100 clusters, then for each
    cluster, we found the image closest to the centroid. Now we have a
    dataset with just 100 labelled instances. We fit the models on these
    centroids.
    
     |       Model         |    Accuracy  |    Time (s)  |    Space (MiB)
     |:-------------------:|:------------:|:------------:|:---------------:
     | Logistic Regression |     70%      |     1.6      |      2114
     |   Neural Network    |     71%      |     3.9      |      2114

-   We then propagated the labels to all the other instances in the same
    cluster.

     |        Model        |   Accuracy   |   Time (s)   |  Space (MiB)
     |:-------------------:|:------------:|:------------:|:---------------:
     | Logistic Regression |     69%      |     653      |      2114
     |   Neural Network    |     69%      |     284      |      2114

-   We then propagated these labels partially because by propagating to
    the full cluster, we have certainly included some outliers. For the
    purpose, we try to find the percentile value with the maximum
    possible accuracy in each model.

    
     |     Percentile          |  Logistic Regression Accuracy  | Neural Network Accuracy
     |:-----------------------:|:------------------------------:|:----------------------:
     |           20            |    70.85%                      |    69.24%
     |           25            |    70.84%                      |    69.05%
     |           50            |    69.82%                      |    68.46%
     |           75            |    69.42%                      |    68.79%
     |-------------------------|--------------------------------|------------------------
     |        Space:           |    2114 MiB                    |    2114 MiB 
     |        Time:            |    1079 s                      |    737 s

     


    \
    We choose the percentile value to be **20** which gave us an
    accuracy of about 71% in Logistic Regression & 69% in Neural
    Network.

## Conclusion

-   We observe that when the models were run on the entire training set,
    the time required was significantly large when compared to the ones
    where we choose a subset of the dataset.

-   The accuracy using semi-supervised learning techniques couldn't be
    improved significantly as we used very small number of labelled
    instances, but in a situation where we get completely unlabelled
    data, these techniques boost the accuracy of our models after we
    have labelled a small but somewhat significant portion of the data
    manually.
