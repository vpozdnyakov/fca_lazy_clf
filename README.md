# Lazy binary classifier based on Formal Concept Analysis

Usually, the work of the classifier can be divided into two steps: the selection of patterns in the training sample (training) and their use in the classification. The lazy classification method differs in that the first step is skipped, and the second step uses the entire training sample, which takes much longer, but can improve the accuracy of the classification (see [report.pdf](report.pdf)).

### Contents of the repository:

* [report.pdf](report.pdf) - report of development
* [fca_lazy_clf](fca_lazy_clf) - source code
* [lazyfca_heart_desease.ipynb](lazyfca_heart_desease.ipynb) - analysis of heart_desease dataset 
* [lazyfca_tic_tac_toe.ipynb](lazyfca_tic_tac_toe.ipynb) - analysis of tic_tac_toe dataset 
* [tic-tac-toe](tic-tac-toe) - tic_tac_toe dataset
* [heart-disease-uci.zip](heart-disease-uci.zip) - heart_desease dataset

### Installation

```sh
$ pip install fca_lazy_clf
```

###  Requirements

The train and test datasets must be represented as ```pandas.DataFrame```. The classifier uses only attributes of types ```numpy.dtype('O')```, ```np.dtype('int64')``` and attributes with 2 any values. Other attributes will not be used. The target attribute must be binary.

### Example

```python
>>> import fca_lazy_clf as fca
>>> import pandas as pd
>>> from sklearn import model_selection

>>> data = pd.read_csv('https://datahub.io/machine-learning/tic-tac-toe-endgame/r/tic-tac-toe.csv')
>>> data.head()

   TL TM TR ML MM MR BL BM BR  class
0  x  x  x  x  o  o  x  o  o   True
1  x  x  x  x  o  o  o  x  o   True
2  x  x  x  x  o  o  o  o  x   True
3  x  x  x  x  o  o  o  b  b   True
4  x  x  x  x  o  o  b  o  b   True

>>> X = data.iloc[:, :-1] # All attributes except the last one
>>> y = data.iloc[:, -1] # Last attribute
>>> X_train, X_test, y_train, y_test\
    = model_selection.train_test_split(X, y, test_size=0.33, random_state=0)

>>> clf = fca.LazyClassifier(threshold=0.000001, bias='negative')
>>> clf.fit(X_train, y_train)
>>> clf.score(X_test, y_test)

0.9716088328075709
```

### Parameters of the classifier

* __bias__ — the decision to make if ```Support+``` is equal to ```Support−```. There are three options: ```'positive'``` (always set a positive class), ```'negative'``` (always set a negative class), and ```'random'``` (set a random class). Read more in the [report.pdf](report.pdf).
* __threshold__ — threshold numeric value from 0 to 1. Read more in the [report.pdf](report.pdf).

* __random__ — ```True``` to enable a mode that uses only a randomly selected portion of the training sample, ```False``` — to disable the mode.
* __sample_share__ — if __random__ mode is used, this parameter sets the percentage of entries from the positive and negative set. Valid values in the range from 0 to 1.
