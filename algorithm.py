# -*- coding: utf-8 -*-
"""
Algorithm helper functions
"""
from sklearn.model_selection import GridSearchCV
from datautils import algorithm_data
import numpy as np

def select_algorithm_grid(kind):
    """
    Select algorithm grid based on y data type
    kind:
    - classification
    - regression
    - clustering
    """
    if kind == "classification":
        return algorithm_data.CLASSIFICATION_ALGORITHM_GRID
    elif kind == "regression":
        return algorithm_data.REGRESSION_ALGORITHM_GRID
    elif kind == "clustering":
        return algorithm_data.CLUSTERING_ALGORITHM_GRID
    
def suggest_algorithm(X, y, kind):
    """
    Suggest best algorithm for given data over common parametrizations
    """
    algorithm_grid = select_algorithm_grid(kind)
    results = get_common_algorithms_best_parametrization(X, y, algorithm_grid)
    score = None
    selected = {}
    for result in results:
        if score is None or score < result.get("score"):
            selected = result
            score = selected.get("score")
    return selected
        
def get_common_algorithms_best_parametrization(X, y, algorithm_grid):
    """
    Get best parametrization for all common algorithms through grid search
    """
    results = []
    for alg, grid_params in algorithm_grid:
        estimator, score, params = grid_search(X, y, alg, grid_params)
        results.append({'estimator': estimator, 'score': score, 'params': params})
    return results

def grid_search(X, y, alg, param_grid, scoring = None, cross_validation = 10, n_jobs = -1):
    """
    Perform grid search over a given algorithm with param grid
    """
    gs = GridSearchCV(estimator = alg,
                      param_grid = param_grid,
                      scoring = scoring,
                      cv = cross_validation,
                      n_jobs = n_jobs
    )
    if y is None:
        gs.fit(X)
    else:
        gs.fit(X, y)
        
    return gs.best_estimator_, gs.best_score_, gs.best_params_
