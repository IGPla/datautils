# -*- coding: utf-8 -*-
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans

COMMON_RANGE = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
COMMON_TOLERANCES = [0.0001, 0.001, 0.01, 0.1]
PENALTIES = ['l1', 'l2']
RANDOM_FOREST_CLASSIFIER_CRITERIONS = ['entropy', 'gini']
RANDOM_FOREST_ESTIMATORS = [1, 10, 100, 1000]
KNEIGHBORS_N = [1, 2, 5, 10, 20]
KNEIGHBORS_WEIGHT = ["uniform", "distance"]
KNEIGHBORS_METRICS = ["minkowski"]
KNEIGHBORS_P = [1, 2, 3, 4, 5]
MLP_SOLVERS = ['lbfgs', 'sgd', 'adam']
MLP_HIDDEN_LAYERS = [(5,), (5, 2), (10, 5, 2)]

CLASSIFICATION_ALGORITHM_GRID = [
    # Logistic regression
    (
        Pipeline([('scl', StandardScaler()),
                  ('clf', LogisticRegression(random_state = 1))]),
        [{'clf__C': COMMON_RANGE,
          'clf__tol': COMMON_TOLERANCES}]
    ),
    (
        Pipeline([('scl', StandardScaler()),
                  ('fs', SelectFromModel(LinearSVC(penalty="l1", C=0.01, dual = False))),
                  ('clf', LogisticRegression(random_state = 1))]),
        [{'clf__C': COMMON_RANGE,
          'clf__tol': COMMON_TOLERANCES}]
    ),
    # SVM
    (
        Pipeline([('scl', StandardScaler()),
                  ('clf', SVC(random_state = 1))]),
        [{'clf__C': COMMON_RANGE,
          'clf__kernel': ['linear']},
         {'clf__C': COMMON_RANGE,
          'clf__kernel': ['rbf'],
          'clf__gamma': COMMON_RANGE}]
    ),
    (
        Pipeline([('scl', StandardScaler()),
                  ('fs', SelectFromModel(LinearSVC(penalty="l1", C=0.01, dual = False))),
                  ('clf', SVC(random_state = 1))]),
        [{'clf__C': COMMON_RANGE,
          'clf__kernel': ['linear']},
         {'clf__C': COMMON_RANGE,
          'clf__kernel': ['rbf'],
          'clf__gamma': COMMON_RANGE}]
    ),
    # Random forest
    (
        Pipeline([('scl', StandardScaler()), # It is not required, but this is for comparing purpose
                  ('clf', RandomForestClassifier(random_state = 1, n_jobs = -1))]),
        [{'clf__criterion': RANDOM_FOREST_CLASSIFIER_CRITERIONS,
          'clf__n_estimators': RANDOM_FOREST_ESTIMATORS}]
    ),
    (
        Pipeline([('scl', StandardScaler()), # It is not required, but this is for comparing purpose
                  ('fs', SelectFromModel(LinearSVC(penalty="l1", C=0.01, dual = False))),
                  ('clf', RandomForestClassifier(random_state = 1, n_jobs = -1))]),
        [{'clf__criterion': RANDOM_FOREST_CLASSIFIER_CRITERIONS,
          'clf__n_estimators': RANDOM_FOREST_ESTIMATORS}]
    ),
    # KNN
    (
        Pipeline([('scl', StandardScaler()),
                  ('clf', KNeighborsClassifier())]),
        [{'clf__n_neighbors': KNEIGHBORS_N,
          'clf__weights': KNEIGHBORS_WEIGHT,
          'clf__metric': KNEIGHBORS_METRICS,
          'clf__p': KNEIGHBORS_P}]
    ),
    (
        Pipeline([('scl', StandardScaler()),
                  ('fs', SelectFromModel(LinearSVC(penalty="l1", C=0.01, dual = False))),
                  ('clf', KNeighborsClassifier())]),
        [{'clf__n_neighbors': KNEIGHBORS_N,
          'clf__weights': KNEIGHBORS_WEIGHT,
          'clf__metric': KNEIGHBORS_METRICS,
          'clf__p': KNEIGHBORS_P}]
    ),
    # Multilayer Perceptron
    (
        Pipeline([('scl', StandardScaler()),
                  ('clf', MLPClassifier(random_state = 1))]),
        [{'clf__solver': MLP_SOLVERS,
          'clf__alpha': COMMON_TOLERANCES,
          'clf__hidden_layer_sizes': MLP_HIDDEN_LAYERS}]
    ),
]

LINEAR_REGRESSION_FIT_INTERCEPTS = [True, False]
LINEAR_REGRESSION_NORMALIZE = [True, False]
RANDOM_FOREST_REGRESSOR_CRITERIONS = ["mse", "mae"]

REGRESSION_ALGORITHM_GRID = [
    # Linear regression
    (
        Pipeline([('scl', StandardScaler()), # It is not required, but this is for comparing purpose
                  ('regr', LinearRegression(n_jobs = -1))]),
        [{'regr__fit_intercept': LINEAR_REGRESSION_FIT_INTERCEPTS,
          'regr__normalize': LINEAR_REGRESSION_NORMALIZE}]
    ),
    # Random forest regressor
    (
        Pipeline([('scl', StandardScaler()), # It is not required, but this is for comparing purpose
                  ('regr', RandomForestRegressor(random_state = 1, n_jobs = -1))]),
        [{'regr__criterion': RANDOM_FOREST_REGRESSOR_CRITERIONS,
          'regr__n_estimators': RANDOM_FOREST_ESTIMATORS}]
    ),
]

KMEANS_CLUSTERS = [2,3,4,5,6,7,8,9,10]

CLUSTERING_ALGORITHM_GRID = [
    # KMeans
    (
        Pipeline([('scl', StandardScaler()),
                  ('clst', KMeans(random_state = 1))]),
        [{'clst__n_clusters': KMEANS_CLUSTERS}]
    ),
]
