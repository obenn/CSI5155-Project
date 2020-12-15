# CSI5155 Final Project
December 16 2020
Oliver Benning


## General remarks

### Coding

The code and results for this project are in the accompanying notebook, project.ipynb. The notebook was made to be as clear and concise as possible, and leaves the verbosity for this document.

Tools used to run this project include: sklearn, numpy, pandas, matplotlib and semisupervised (containing keras and torch). The project was created in the Anaconda Python 3.8 environment. The notebook takes about 20 hours to run on a modern machine with GPU acceleration.

## Part A: Supervised learning

### Feature engineering

The goal here is to ensure all features are numerical so the data is compatible with more ML algorithms. We utilize categorical and one-hot encodings to encode categorical data, paying attention not to overload our data with dimentionality from one-hot encoding, nor to lose too much information either.

Features that did not provide qualitative information were dropped, this included encounter id and patient number. This is because they're assigned arbitrarily, and provide nothing qualitative to learn from.

The only qualitative column that was dropped was weight. It was chosen to drop the column as 98% of observations was missing and I did not see a clear path to impute.

Columns that were categorical but not ordinal, such as admission type (which has numerical values but only because they are encoded with ids) were one-hot encoded. Some categorical columns contained multiple no information values, like "NULL", "not available", and "not mapped", so these were fused into one category.

The 3 diagnosis columns presented a risk of making the data too sparse one-hot encoded as-is, due to the large amount of features, i.e. medical diagnosis ids within. The associated paper from the data outlined reasonable category groupings for these ids, so they were grouped according to the paper in order to reduce dimensionality.

Categorical columns with an ordering were mapped to numerical values representing the ordering. Binary columns were mapped to a -1, 0, 1 mapping, where 0 represented unknown if present. Ordinal columns with more than two features were mapped to 0, 1, 2... with 0 representing an unknown or null case if present.

With the exception of weight, the dataset barely had any missing values so no imputation was required. Only categorical features had unknown values, which just assumed the 0 category.

Assumptions:
- I decided not to group the medical_speciality column despite it containing 73 categories as I was not familiar with the different categories of medicine, the size of the tail set of features with <10 observations also was not unreasonable. In a real case I would do more research or ask a domain expert for a grouping suggestion.
- I decided not to regularize the domain of features as none of the numerical features had too large a domain, I also read up and found that most regularization sensitive algorithms do regularization internally.
- I made a judgement call dropping weight but I would revisit it more in depth to see if the 2% could in fact generalize the 98% well.


## Task 1

The algorithms chosen here were Decision Tree (DT), Naive Bais (NB) and k Nearest Neighbours (kNN). Bagging (BA) was also tried using the GradientBoostingClassifier. These algorithms were chosen due to their native multi-class support, as well as to get a good mix of variety in the learners. Boosting was chosen to see if it can improve on the single learner results.

For every train-test prediction we make we use 10-fold cross validation for better results, since we can afford the training time.

Starting with the single learners. For each algorithm three ROC curves were generated for the 3 combinations of the 3 class features in readmitted. The k for kNN was selected using test predictions on a holdout set and choosing the least mean error.

The prediction here is more of two nested binary predictions. We care most about 0 vs. {1,2}, i.e. whether or not they were readmitted at all, and then 1 vs 2 to show how long it took. We observe O.K. results for the 0-1 and 0-2 paired ROC curves, but notable very low scores for 1-2 curve in the single learner algorithms. This suggests that readmittal vs no readmittal performance may be passable, but differentiating over or under 30 days is likely not possible.

Next, a confusion matrix was plotted for each algorithm along with statistics about predictions, namely recall (TPR), specificity (TNR), precision, and F-measure.

First, we make sure the algorithms are statistically different. Since 