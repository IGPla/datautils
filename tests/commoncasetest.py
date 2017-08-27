# -*- coding: utf-8 -*-
import unittest
import os
from datautils import commoncase
from datautils import algorithm_data
from sklearn.pipeline import Pipeline
from datautils.tests import basetest
class CommonCaseTestCase(basetest.BaseTestCase):
    def setUp(self):
        super(CommonCaseTestCase, self).setUp()
        self.train_filename = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/IRIS.csv")
        self.target_column_classification = "Species"
        self.target_column_regression = "SepalLengthCm"
        self.identifier_column = "Id"

    def test_common_data_preparation(self):
        """
        Test common_data_preparation function
        """
        X, y, identifiers, df = commoncase.common_data_preparation(self.train_filename,
                                                                   self.target_column_classification,
                                                                   self.identifier_column)
        self.assertEqual(X.shape, (150, 4))
        self.assertEqual(y.shape, (150,))
        columns = X.columns.values.tolist()
        columns.sort()
        desired_columns = ['PetalLengthCm', 'PetalWidthCm', 'SepalLengthCm', 'SepalWidthCm']
        desired_columns.sort()
        self.assertEqual(columns, desired_columns)

    def test_quick_test_classification(self):
        """
        Test quick_test function with classification
        """
        algorithm = commoncase.quick_test(self.train_filename,
                                          self.target_column_classification,
                                          self.identifier_column,
                                          quick_and_dirty = True,
                                          verbose = False,
                                          kind = "classification")
        
        self.assertEqual(type(algorithm), Pipeline)
        self.assertTrue("clf" in algorithm.named_steps.keys())

        algorithm = commoncase.quick_test(self.train_filename,
                                          self.target_column_classification,
                                          self.identifier_column,
                                          quick_and_dirty = False,
                                          verbose = False,
                                          kind = "classification")
        
        self.assertEqual(type(algorithm), Pipeline)
        self.assertTrue("clf" in algorithm.named_steps.keys())

    def test_quick_test_regression(self):
        """
        Test quick_test function with regression
        """
        algorithm = commoncase.quick_test(self.train_filename,
                                          self.target_column_regression,
                                          self.identifier_column,
                                          quick_and_dirty = True,
                                          verbose = False,
                                          kind = "regression")
        self.assertEqual(type(algorithm), Pipeline)
        self.assertTrue("regr" in algorithm.named_steps.keys())

        algorithm = commoncase.quick_test(self.train_filename,
                                          self.target_column_regression,
                                          self.identifier_column,
                                          quick_and_dirty = False,
                                          verbose = False,
                                          kind = "regression")
        self.assertEqual(type(algorithm), Pipeline)
        self.assertTrue("regr" in algorithm.named_steps.keys())

    def test_quick_test_clustering(self):
        """
        Test quick_test function with clustering
        """
        algorithm = commoncase.quick_test(self.train_filename,
                                          None,
                                          self.identifier_column,
                                          quick_and_dirty = True,
                                          verbose = False,
                                          kind = "clustering")
        self.assertEqual(type(algorithm), Pipeline)
        self.assertTrue("clst" in algorithm.named_steps.keys())
        
        algorithm = commoncase.quick_test(self.train_filename,
                                          None,
                                          self.identifier_column,
                                          quick_and_dirty = False,
                                          verbose = False,
                                          kind = "clustering")
        self.assertEqual(type(algorithm), Pipeline)
        self.assertTrue("clst" in algorithm.named_steps.keys())

        
if __name__ == '__main__':
    unittest.main()
    
