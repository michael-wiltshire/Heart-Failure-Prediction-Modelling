import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

class MappingTransformer(BaseEstimator, TransformerMixin):

  def __init__(self, mapping_column, mapping_dict:dict):  
    self.mapping_dict = mapping_dict
    self.mapping_column = mapping_column  #column to focus on

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    assert isinstance(X, pd.core.frame.DataFrame), f'MappingTransformer.transform expected Dataframe but got {type(X)} instead.'
    assert self.mapping_column in X.columns.to_list(), f'MappingTransformer.transform unknown column {self.mapping_column}'
    X_ = X.copy()
    X_[self.mapping_column].replace(self.mapping_dict, inplace=True)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

class OHETransformer(BaseEstimator, TransformerMixin):
  def __init__(self, target_column, dummy_na=False, drop_first=True):  
    self.target_column = target_column
    self.dummy_na = dummy_na
    self.drop_first = drop_first
  
  #fill in the rest below
  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    X_ = X.copy()
    X_= pd.get_dummies(X_,
                               prefix=self.target_column,    #your choice
                               prefix_sep='_',     #your choice
                               columns=[self.target_column],
                               dummy_na=self.dummy_na,    #will try to impute later so leave NaNs in place
                               drop_first=self.drop_first    #will drop Belfast and infer it
                               )
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result

class DropColumnsTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, column_list, action='drop'):
    assert action in ['keep', 'drop'], f'DropColumnsTransformer action {action} not in ["keep", "drop"]'
    assert isinstance(column_list, list), f'DropColumnsTransformer expected list but saw {type(column_list)}'
    self.column_list = column_list
    self.action = action

  def fit(self, X, y = None):
    print("Warning: MappingTransformer.fit does nothing.")
    return X

  def transform(self, X):
    X_ = X.copy()
    if self.action=='keep': 
      X_ =X_[self.column_list]
    else: 
      X_ =X_.drop(columns=self.column_list)
    return X_

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result





class PearsonTransformer(BaseEstimator, TransformerMixin):
  def __init__(self, threshold):
    self.threshold = threshold

  #define methods below
  def transform(self, X):
    X_ = X
    df_corr = X
    df_corr = transformed_df.corr(method='pearson')

    threshold = .4
    masked_df = df_corr.abs() > threshold

    upper_mask = masked_df
    m,n = upper_mask.shape
    upper_mask[:] = np.where(np.arange(m)[:,None] >= np.arange(n),False,upper_mask)
    upper_mask = upper_mask.astype(bool)

    correlated_columns = upper_mask.columns[      
    (upper_mask == True)        # mask 
    .any(axis=0)].tolist()

    new_df = upper_mask.drop(columns=correlated_columns)
    

    return new_df

  def fit_transform(self, X, y = None):
    result = self.transform(X)
    return result
  
  
