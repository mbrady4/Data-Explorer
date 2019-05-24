#!/usr/bin/env python 
"""
lambdata - A collection of Data Science helper functions
"""

import pandas as pd
import numpy as np 
from . import example_module

Y = example_module.increment(example_module.X)
TEST = pd.DataFrame(np.ones(10))