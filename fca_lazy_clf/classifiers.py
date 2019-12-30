from sklearn.base import BaseEstimator, ClassifierMixin
import random
import numpy as np
import pandas as pd

class LazyClassifier(BaseEstimator, ClassifierMixin):
    def __init__(
        self, threshold=0.5, 
        random=False, sample_share=0.5, 
        bias='random', random_seed=None):
        
        self.threshold = threshold
        self.random = random
        self.sample_share = sample_share
        self.bias = bias
        self.random_seed = random_seed
        self.binary_mapping = dict()
    
    def fit(self, X, y):
        pd.options.mode.chained_assignment = None
        X = self.scaled_X(X)
        y = self.scaled_y(y)
        self.positive_sample = X[y == 1]
        self.negative_sample = X[y == 0]

        if self.random:
            sample_size = int(self.sample_share * self.positive_sample.shape[0])
            self.positive_sample = self.positive_sample.sample(
                    n=sample_size, random_state=self.random_seed)
            self.negative_sample = self.negative_sample.sample(
                    n=sample_size, random_state=self.random_seed)
        
        self.positive_obj = {}
        self.negative_obj = {}
        pos = self.positive_sample
        neg = self.negative_sample
        for i_col in X.columns:
            self.positive_obj[i_col] = pos[i_col][pos[i_col] == 1].index
            self.negative_obj[i_col] = neg[i_col][neg[i_col] == 1].index
        
    def predict(self, X):
        pd.options.mode.chained_assignment = None
        random.seed(self.random_seed)
        X = self.scaled_X(X)
        predictions = []
        for i_obj in range(X.shape[0]):
            i_extent = self.extent(X.iloc[i_obj])
            support_pos  = self.calculate_support(i_extent, 'positive')
            support_neg  = self.calculate_support(i_extent, 'negative')
            
            if support_neg == support_pos:
                if self.bias == 'random':
                    prediction = random.choice([True, False])
                elif self.bias == 'positive':
                    prediction = True
                else:
                    prediction = False
            else: 
                prediction = support_pos > support_neg
            predictions.append(self.binary_mapping[prediction])
        return predictions
    
    def scaled_X(self, X_dataset):
        intervals = 5
        for i_col in X_dataset.columns:
            values = list(X_dataset[i_col].unique())

            if len(values) == 2 and 0 in values and 1 in values:
                continue
            elif len(values) == 1 and (0 in values or 1 in values):
                continue
            
            elif len(values) <= 2 or X_dataset[i_col].dtypes == np.dtype('O'):
                values = sorted(list(X_dataset[i_col].unique()))
                for i_val in values:
                    X_dataset['{}_{}'.format(i_col, i_val)]\
                        = (X_dataset[i_col] == i_val).astype(int)
            
            elif X_dataset[i_col].dtype == np.dtype('int64'):
                min_val = X_dataset[i_col].min()
                max_val = X_dataset[i_col].max()
                gap = max_val - min_val
                start = min_val + gap / intervals
                finish = max_val - gap / intervals
                k = 0
                for i in np.linspace(start, finish, intervals):
                    X_dataset['{}_{}'.format(i_col, k)]\
                        = (X_dataset[i_col] >= i).astype(int)
                    k += 1

            X_dataset.drop([i_col], axis=1, inplace = True)
        return X_dataset
    
    def scaled_y(self, y_series):
        values = sorted(y_series.unique())
        if len(values) != 2:
            raise Exception('Only a binary target feature is possible')
        self.binary_mapping[False] = values[0]
        self.binary_mapping[True] = values[1]
        return (y_series == values[1]).astype(int)

    def calculate_support(self, obj_ext, base):

        base_sample = (self.positive_sample if base == 'positive' 
                else self.negative_sample)
        review_sample = (self.negative_sample if base == 'positive' 
                else self.positive_sample)
        review_obj = (self.negative_obj if base == 'positive' 
                else self.positive_obj)

        res = 0
        for _, i_obj in base_sample.iterrows():
            i_inters = self.intersection(
                obj_ext, self.extent(i_obj))
            support_card = 0
            if i_inters: 
                support = review_obj[i_inters[0]]
                for i_col in i_inters:
                    support = self.intersection(support, review_obj[i_col])
                    if not support: break
                support_card = len(support) / review_sample.shape[0]
                if support_card < self.threshold:
                    res += len(i_inters) / len(obj_ext)
        
        res = res / base_sample.shape[0]
        return res

    def extent(self, series):
        return series[series == 1].index.tolist()

    def intersection(self, L, R):
        return [val for val in L if val in R]

    def belongs(self, sub, base):
        return len(self.intersection(sub, base)) == len(sub)