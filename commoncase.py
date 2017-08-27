# -*- coding: utf-8 -*-
"""
Common data analysis cases, encapsulated, for quick diagnosis
"""
import numpy as np
import pandas as pd
from datautils import preprocess, algorithm, algorithm_data, dataio

def common_data_preparation(filename, target_column, index_column, with_target = True):
    """
    Common data preprocess and preparation
    """
    df = pd.read_csv(filename)
    X, y, identifiers = preprocess.build_vectors(df = df,
                                                 target_column = target_column,
                                                 index_column = index_column,
                                                 with_target = with_target)
    preprocess.replace_all_nulls(X)
    preprocess.encode_labels(X)
    return X, y, identifiers, df

def quick_test(train_filename, target_column, index_column, verbose = True, quick_and_dirty = True, kind = "classification"):
    """
    Define a quick test against provided data. Provided files must be csv
    target_column: if not provided, it will be a clustering problem
    """
    with_target = target_column is not None
    X_train, y_train, ids_train, df_train = common_data_preparation(train_filename,
                                                                    target_column,
                                                                    index_column,
                                                                    with_target)
    if with_target:
        X_train, X_test, y_train, y_test = preprocess.split_dataframe(X_train, y_train)
    else:
        X_train, X_test = preprocess.split_dataframe(X_train)
        y_train = y_test = None
        
    if quick_and_dirty:
        alg_data = algorithm.select_algorithm_grid(kind)[0]
        alg, score, parametrization = algorithm.grid_search(X_train, y_train, alg_data[0], alg_data[1])
    else:
        bundle = algorithm.suggest_algorithm(X_train, y_train, kind)
        alg, parametrization = bundle.get("estimator"), bundle.get("params")
    if verbose:
        print("Best algorithm selected: %s" % str(alg))
        print("Best algorithm parametrization: %s" % str(parametrization))
        if with_target:
            print("Train score: %.3f" % alg.score(X_train, y_train))
            print("Test score: %.3f" % alg.score(X_test, y_test))
        else:
            print("Train score: %.3f" % alg.score(X_train))
            print("Test score: %.3f" % alg.score(X_test))
    return alg

def quick_output(train_filename, test_filename, target_column, index_column, test_data_has_target = False, quick_and_dirty = True, verbose = False, save_model = False, kind = "classification"):
    """
    Set up a quick and dirty output for a given problem. 
    Generate a model, predict against test data and write output
    Useful to get a quick answer on a given classification problem

    test_data_has_target: True if test dataset has target column
    target_column: if not provided, it will be a clustering problem
    """
    with_target = target_column is not None
    alg = quick_test(train_filename, target_column, index_column, verbose, quick_and_dirty, kind = kind)
    X_test, y_test, ids_test, df_test = common_data_preparation(test_filename,
                                                                target_column,
                                                                index_column,
                                                                test_data_has_target and with_target)
    y_test = alg.predict(X_test)
    dataio.write_output(y_test, target_column, ids_test, index_column)
    if save_model:
        dataio.write_model(alg)

