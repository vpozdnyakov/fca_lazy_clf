# Lazy binary classifier based on Formal Concept Analysis

The lazy classification method differs in that the classifier uses the entire training sample without advance training, which takes much longer, but can increase the accuracy of the classification (see report.pdf).

### Installation

```sh
$ pip install fca_lazy_clf
```

###  Usage

The classifier uses only attributes of types numpy.dtype('O'), np.dtype('int64') and attributes with 2 any values. Other attributes are not taken into account. The target attribute must be binary. The train and test data sets must be represented as pandas.DataFrame.

```sh
import pandas as pd
import fca_lazy_clf as fca
from sklearn import model_selection

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