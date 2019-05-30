#!/usr/bin/env python

import pandas as pd
import numpy as np

from explorer import explorer

import unittest


class ExplorerTest(unittest.TestCase):
    
    def test_init(self):
        ''' Test Initialization Class'''
        # Create a frame with cardinality of 3 and 4
        df = pd.DataFrame({'a': [1, 3, 4, 5],
                     'b': [6, 4, 6, 7]
                     })
        target = pd.Series([0,1,1,0])
        test = explorer(df, target)
        self.assertEqual(test.df.columns[0], df.columns[0])
        self.assertEqual(test.df.columns[-1], df.columns[-1])
        self.assertEqual(test.target[0], target[0])

    def test_data_dict(self):
        ''' Test Data Dict Class'''
        # Create a frame with cardinality of 3 and 4
        df = pd.DataFrame({'a': [1, 3, 4, 5],
                     'b': [6, 4, 6, 7]
                     })
        target = pd.Series([0,1,1,0])
        test = explorer(df, target)
        dict = test.data_dict()
        self.assertEqual(len(dict.columns), 10)

if __name__ == '__main__':
    unittest.main()
