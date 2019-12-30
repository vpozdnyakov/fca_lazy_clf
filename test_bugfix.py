import fca_lazy_clf as fca
import pandas as pd
from sklearn import metrics

def test_creation():
    assert type(fca.LazyClassifier()) == fca.LazyClassifier

# https://github.com/vpozdnyakov/FCALazyClassifier/issues/2
def test_bugfix_2():
    clf = fca.LazyClassifier(
        threshold=0.000001, bias='false', 
        random=True, sample_share=0.3, random_seed=1)

    train_data = scale(pd.read_csv('tic-tac-toe/train{}.csv'.format(1)))
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    clf.fit(X_train, y_train)

    test_data = scale(pd.read_csv('tic-tac-toe/test{}.csv'.format(1)))
    X_test = test_data.iloc[-10:, :-1]
    y_test = test_data.iloc[-10:, -1]

    y_pred = clf.predict(X_test)

    assert metrics.accuracy_score(y_test, y_pred) == 1.0

def scale(dataset):
    for i in range(9):
        str_i = str(i + 1)
        dataset['v' + str_i] = (dataset['V' + str_i] == 'x').astype(int)
    dataset['v10'] = (dataset['V10'] == 'positive').astype(int)
    dataset.drop(['V' + str(i+1) for i in range(10)], axis=1, inplace = True)
    return dataset