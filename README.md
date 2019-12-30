# Lazy binary classifier based on Formal Concept Analysis

The lazy classification method differs in that the classifier uses the entire training sample without advance training, which takes much longer, but can increase the accuracy of the classification (see report.pdf).

### Contents of the repository:
* report.pdf - report of the development
* fca_lazy_clf - source code
* lazyfca_heart_desease.ipynb - analysis of heart_desease data set 
* lazyfca_tic_tac_toe.ipynb - analysis of tic_tac_toe data set 
* tic-tac-toe - tic_tac_toe dataset
* heart-disease-uci.zip - heart_desease dataset

### Installation

```sh
$ pip install fca_lazy_clf
```

###  Usage

The classifier uses only attributes of types numpy.dtype('O'), np.dtype('int64') and attributes with 2 any values. Other attributes are not taken into account. The target attribute must be binary. The train and test data sets must be represented as pandas.DataFrame.

```python
import fca_lazy_clf as fca
import pandas as pd
from sklearn import model_selection, metrics

data = pd.read_csv('https://datahub.io/machine-learning/tic-tac-toe-endgame/r/tic-tac-toe.csv')

X = data.iloc[:, :-1] # All attributes except the last one
y = data.iloc[:, -1] # Last attribute
X_train, X_test, y_train, y_test\
    = model_selection.train_test_split(X, y, test_size=0.33, random_state=0)

clf = fca.LazyClassifier(threshold=0.000001, bias='false')
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print(metrics.accuracy_score(y_test, y_pred))
>>> 0.9716088328075709
```

### Parameters of the classifier
* __bias__ — the decision to make if Support+ = Support−. There are three options: ```True``` (always set a positive class), ```False``` (always set a negative class), and ```'random'``` (set a random class). Read more in the report.
* __threshold__ — threshold numeric value from 0 to 1. Read more in the report.

* __random__ — ```True``` to enable a mode that uses only a randomly selected portion of the training sample, ```False``` — to disable the mode.
* __sample_share__ — if __random__ mode is used, this parameter sets the percentage of entries from the positive and negative set. Valid values in the range [0, 1].
