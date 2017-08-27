# -*- coding: utf-8 -*-
"""
Preprocess utilities
"""
import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

def one_hot_encoder(df, column, null_replacement = "most_frequent"):
    """
    Takes all values from df[column] and create new columns for each different value. Perform one hot encoder into them
    null_replacement param accepts the following values:
    - most_frequent: replace with most frequent value
    - another value: replace with provided value
    """
    replace_null(df, column, null_replacement)
    vals = df[column].unique()
    for val in vals:
        _val = "%s_%s" % (column, val)
        df.loc[df[column] == val, _val] = 1
        df.loc[df[column] != val, _val] = 0

def replace_null(df, column, null_replacement = "most_frequent", fallback = -1):
    """
    Replace null values
    fallback is used as a workaround when all values in column are null
    available replacement strategies:
    - most_frequent: use most frequent value to fill nans
    """
    if null_replacement == "most_frequent":
        try:
            null_replacement = df[column].value_counts().keys()[0]
        except:
            null_replacement = fallback
    df.loc[pd.isnull(df[column]), column] = null_replacement

def replace_all_nulls(df, null_replacement = "most_frequent", fallback = -1):
    """
    Replace null for all columns
    """
    for key in df.keys():
        replace_null(df, key, null_replacement, fallback)

def drop_na_data(df, columns = True, rows = True):
    """
    Drop columns and/or rows with all nulls
    """
    if columns:
        df = df.dropna(axis = 1)
    if rows:
        df = df.dropna(axis = 0)
    return df
        
def get_features_by_importance(df, X_columns, y_column, alg = None):
    """
    Return a list of features, sorted by importance on algorithm training
    if alg is not provided, LinearSVC will be used
    """
    if not alg:
        alg = LinearSVC(penalty="l1", C=0.01, dual = False)
    alg.fit(df[X_columns], df[y_column])
    model = SelectFromModel(alg, prefit = True)
    return model, model.transform(df[X_columns])

def build_vectors(df,
                  with_target = True,
                  target_column = "target",
                  index_column = "index",
                  features = None):
    """
    Build features vector and target vector. Optionally, return identifiers too
    If features vector is provided, it will be used instead of all columns for X
    """
    columns = df.columns
    y = []
    if with_target:
        y = df.loc[:, target_column]
        columns = columns.drop([index_column, target_column])
    else:
        columns = columns.drop([index_column])
    identifiers = df.loc[:, index_column]
    if features:
        columns = features
    X = df.loc[:, columns]
    return X, y, identifiers

def encode_labels(df):
    """
    Encode labels with labelencoder
    """
    for column in df.columns:
        if df[column].dtype.type == np.object_:
            le = LabelEncoder()
            df.loc[:, column] = le.fit_transform(df.loc[:, column])

def split_dataframe(*args):
    """
    Split dataframe into train and test
    """
    return train_test_split(*args, test_size = 0.2, random_state = 1)
