# **Classification Models for Bank Marketing Dataset**

## Author
- Rishika Tibrewal
- Shreyansh Rastogi

## Overview

The **Bank Marketing Dataset** from the UCI Machine Learning Repository
is related with direct marketing campaigns (phone calls) of a Portuguese
banking institution. The marketing campaigns were based on phone calls.
Often, more than one contact to the same client was required, in order
to access if the product (bank term deposit) would be ('yes') or not
('no') subscribed.\
\
**Some of the variables involved:**

-   age (numeric)

-   job : type of job (categorical)

-   marital : marital status (categorical)

-   education (categorical)

-   default: has credit in default? (categorical)

-   housing: has housing loan? (categorical)

-   loan: has personal loan? (categorical)

-   contact: contact communication type (categorical)

-   month: last contact month of year (categorical)

-   day_of_week: last contact day of the week (categorical)

-   duration: last contact duration, in seconds (numeric)

-   campaign: number of contacts performed during this campaign and for
    this client (numeric)

-   pdays: number of days that passed by after the client was last
    contacted from a previous campaign (numeric; 999 means client was
    not previously contacted)

[**Click here**](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
to view the full details of the variables in the source page.

## Objective

The goal is to predict if the client will subscribe a term deposit
(variable **y**) using classification models.

## Methodology

We have built three classifiers for this dataset: **Decision tree**,
**Naïve Bayes classifie**r, and **Random Forest**, and hence compared
their performances on this dataset using a suitable metric (here,
**Recall** for **yes**). Cautious Classifier here classifies the
interested customers uninterested, which in turn brings down the bank's
business, so we have to decrease FN and hence increase recall.
Increasing recall makes sense as we have to actually increase the
percentage of true positive cases, among all the actual positive ones.
$$Recall = \frac{TP}{TP+FN}\$$\
The libraries used are **Pandas, NumPy, Matplotlib, Sklearn, Imblearn,
time, seaborn, warnings** and **memory_profiler**.

## Data Preprocessing

-   Starting with Exploratory Data Analysis on the dataset, we try to
    visualise the categorical columns in all the three models. In
    **Naïve Bayes**, we plot the numerical columns as well.

-   We dropped columns **duration** (as the duration is not known before
    a call is performed), and **default** (high proportion of the data
    here is unknown with a very negligible proportion of yes). In
    **Naïve Bayes**, we drop the rows with entry as **unknown** (as
    unknown occurs in many columns, and that will in turn increase its
    occurence probability).

-   We used Label encoding for ordinal columns and One Hot encoding for
    nominal columns in **Decision tree** and **Random Forest**. In
    **Naïve Bayes**, we standardised the numerical columns and performed
    ordinal encoding on categorical columns (we used Pipeline structure
    for this).

-   We set aside 30% of the dataset as the test data and separated the
    set of attributes and target variable **y** both in training data
    and test data.

-   In **Naïve Bayes:**, we replaced all the outliers with some
    appropriate values. The value '999' in **pdays** (meaning that the
    client was not previously contacted) was replaced by 0, and the
    values in **campaign** which are greater than 20 are replaced by the
    mean value.

## Implementation

-   **Decision Tree:** Hyperparameter tuning is performed to find the
    optimum max_depth and min_samples_split in the tree. Further since,
    data is highly imbalanced (**no** is a majority here, and **yes** is
    a minority), we compute and assign weights to the classes, giving
    more weight to **yes**. The score of this model is about 82.83%.

-   **Random Forest:** The Random Forest is applied on the sampled rows
    (using random over sampling with sampling strategy of 0.5) of
    training dataset. The sampling strategy would ensure that the
    minority class was oversampled to have half the number of examples
    as the majority class, for this binary classification problem (as
    the data is imbalanced). The score of this model is about 86.63%.

-   **Naïve Bayes:** We fitted Gaussian Naïve Bayes on numerical columns
    and Categorical Naïve Bayes on categorical columns. Next we computed
    the probability of obtaining this data from each of these Naïve
    Bayes models and merged them. We fitted the Gaussian Naïve Bayes on
    these probabilities. The score of this model is about 87.07%.

## Comparison of Measures

 |  **Performance Measure** |  **Decision Tree**|  **Naïve Bayes** |  **Random Forest**
 |:------------------------:|:-----------------:|:----------------:|:-------------------:
 |        Precision         |       36%         |      43%         |       43%
 |        Accuracy          |       83%         |      87%         |       87%
 |         Recall           |       64%         |      46%         |       58%
 |        Time(ms)          |       346         |      306         |      1260
 |       Space(MiB)         |       348         |      294         |       316

[**Click here**](https://drive.google.com/drive/folders/1wmGecimy3Upl26joHUC1R1TUSrMh-uii?usp=sharing)
to view the output.

## Conclusion

-   Dataset has highly imbalanced data of **'yes'** & **'no'**. Dataset
    has outliers in the columns **campaign** & **pdays** which we
    replaced by appropriate values.

-   For **Decision tree** and **Random Forest**, we had to perform
    precision-recall trade-off. We tried to increase the **recall** of
    **yes** without decreasing accuracy and precision of the model much.

-   Clearly, in terms of **accuracy**, both **Random Forest and Naïve
    Bayes** are performing better than **Decision tree**. In terms of
    **recall** of **yes**, we can choose **Decision tree** to be our
    preferred model in our case.
