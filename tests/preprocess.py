# -*- coding: utf-8 -*-
import unittest
import pandas as pd
import numpy as np
from datautils import preprocess
from datautils.tests import basetest

class PreprocessTestCase(basetest.BaseTestCase):
    def setUp(self):
        super(PreprocessTestCase, self).setUp()
        self.build_dataframe()
        
    def build_dataframe(self):
        """
        Build test dataframe
        """
        self.df = pd.DataFrame({'col1': [None, 1, 1, 3],
                                'col2': [1, 2, 3, 4],
                                'col3': [1, 3, 3, None],
                                'col4': ["a", "b", "c", "d"]})
        
    def test_one_hot_encoder(self):
        """
        test one_hot_encoder function
        """
        base_columns = ['col1', 'col2', 'col3', 'col4']
        old_columns = self.df.columns.values.tolist()
        base_columns.sort()
        old_columns.sort()
        self.assertEqual(old_columns, base_columns)
        preprocess.one_hot_encoder(self.df, "col4")
        base_columns += ['col4_a', 'col4_b', 'col4_c', 'col4_d']
        new_columns = self.df.columns.values.tolist()
        base_columns.sort()
        new_columns.sort()
        self.assertEqual(new_columns, base_columns)
        
    def test_replace_null(self):
        """
        test replace_null function
        """
        preprocess.replace_null(self.df, "col1")
        NAN_PLACEHOLDER = "NAN PLACEHOLDER"
        results = self.df.values.tolist()
        for group in results:
            for index, item in enumerate(group):
                try:
                    if np.isnan(item):
                        group[index] = NAN_PLACEHOLDER
                except:
                    pass
        self.assertEqual(results, [[1.0, 1, 1.0, 'a'], [1.0, 2, 3.0, 'b'], [1.0, 3, 3.0, 'c'], [3.0, 4, NAN_PLACEHOLDER, 'd']])

    def test_replace_all_nulls(self):
        """
        test replace_all_nulls function
        """
        preprocess.replace_all_nulls(self.df)
        self.assertEqual(self.df.values.tolist(), [[1.0, 1, 1.0, 'a'], [1.0, 2, 3.0, 'b'], [1.0, 3, 3.0, 'c'], [3.0, 4, 3.0, 'd']])

    def test_drop_na_data(self):
        """
        test drop_na_data function
        """
        df = preprocess.drop_na_data(self.df, True, False)
        self.assertEqual(df.values.tolist(), self.df.loc[:, ["col2", "col4"]].values.tolist())
        df = preprocess.drop_na_data(self.df, False, True)
        self.assertEqual(df.values.tolist(), self.df.loc[[1, 2], :].values.tolist())

    def test_get_features_by_importance(self):
        """
        test get_features_by_importance function
        """
        pass

    def test_build_vectors(self):
        """
        test build_vectors function
        """
        X, y, identifiers = preprocess.build_vectors(self.df, True, "col4", "col2")
        for part1, part2 in [
                (self.df.loc[:, ['col1', 'col3']], X),
                (self.df["col4"], y),
                (self.df["col2"], identifiers)
        ]:    
            self.assertEqual(part1.to_json(), part2.to_json())

    def test_encode_labels(self):
        """
        test encode_labels function
        """
        self.assertEqual(self.df.col4.dtype.type, np.object_)
        preprocess.encode_labels(self.df)
        self.assertEqual(self.df.col4.dtype.type, np.int64)

    def test_split_dataframe(self):
        """
        test split_dataframe function
        """
        X_train, X_test = preprocess.split_dataframe(self.df)
        self.assertEqual(X_train.shape[1], self.df.columns.size)
        self.assertEqual(X_test.shape[1], self.df.columns.size)
        self.assertTrue(X_train.shape[0] > 0)
        self.assertTrue(X_test.shape[0] > 0)
        
if __name__ == "__main__":
    unittest.main()
