# -*- coding: utf-8 -*-
import unittest
import warnings
import os
import sys

class BaseTestCase(unittest.TestCase):
    def setUp(self):
        def warn(*args, **kwargs):
            pass
        warnings.warn = warn
 
