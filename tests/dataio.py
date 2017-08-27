# -*- coding: utf-8 -*-
import unittest
import pandas as pd
from sklearn.linear_model import LogisticRegression
from datautils import dataio
from datautils.tests import basetest

class DataioTestCase(basetest.BaseTestCase):
    def setUp(self):
        super(DataioTestCase, self).setUp()

    def test_write_output(self):
        """
        Test write_output function
        """
        prediction = ["a","b","c","d"]
        prediction_label = "Test"
        identifiers = [1,2,3,4]
        identifiers_label = "Ids"
        filepath = "/tmp/test.csv"
        dataio.write_output(prediction, prediction_label, identifiers, identifiers_label, filepath, verbose = False)
        df = pd.read_csv(filepath)
        self.assertEqual(df.loc[:, "Test"].values.tolist(), prediction)
        
    def test_write_model_load_model(self):
        """
        Test write_model and load_model functions
        """
        model = LogisticRegression(C = 0.001)
        filepath = "/tmp/test.pickle"
        dataio.write_model(model, filepath, verbose = False)
        loadedmodel = dataio.load_model(filepath)
        self.assertEqual(model.C, loadedmodel.C)
    
if __name__ == "__main__":
    unittest.main()
