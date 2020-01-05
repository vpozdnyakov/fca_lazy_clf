import setuptools

setuptools.setup(
  name='fca_lazy_clf',
  packages=['fca_lazy_clf'],
  version='0.3',
  license='MIT',
  description='Lazy binary classifier based on Formal Concept Analysis',
  long_description="""
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

  >>> clf = fca.LazyClassifier(threshold=0.000001, bias='false')
  >>> clf.fit(X_train, y_train)
  >>> clf.score(X_test, y_test)

  0.9716088328075709
  ```
  """,
  long_description_content_type="text/markdown",
  author='Vitaliy Pozdnyakov',
  author_email='pozdnyakov.vitaliy@yandex.ru',
  url='https://github.com/vpozdnyakov/fca_lazy_clf',
  keywords=['fca', 'formal-concept-analysis', 'lazy-learning', 'binary-classification'],
  install_requires=[
          'pandas',
          'numpy',
          'sklearn',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',
    'Intended Audience :: Developers',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
  ],
)