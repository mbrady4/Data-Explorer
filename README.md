# Data Explorer
> A collection of functions to accelerate exploration of a Dataset

The following functions are available: 
- **data_dict(df, target)**: Create a DataFrame intended to orient the user to a dataset.
- **get_cardinality(df, include_numeric)**: Determine the cardinality of each column. 
- **corr_with_target(features, target)**: Determine correlations of numeric columns with a specified target. 
- **detect_suspicious(df, custom_vals)**: Searches a DataFrame for values that should possibly be considered np.nan in a given DataFrame.
- **detect_outliers(df, method, threshold)**: Identify outliers in numeric columns using the specified outlier detection method.
- **outlier_modified_z_score(df, threshold)**: Identify outliers in numeric columns using the modified z score range method. A modified z score uses the median rather than the mean.
- **outlier_z_score(df, threshold)**: Identify outliers in numeric columns using the z-score method. 
- **outlier_iqr(df, threshold)**: Identify outliers in numeric columns using the interquartile range method.

## Installation: 
```pip install -i https://test.pypi.org/simple/ data-explorer```

## Implementation: 
The following code block returns a useful summary that can inform further data exploration, cleaning, and engineering: 
```import pandas as pd
# Assumes the data-explorer package has already been installed
from lambdata_mbrady4 import feature_explorer
king = pd.read_csv('https://raw.githubusercontent.com/ryanleeallred/datasets/master/kc_house_data.csv')
feature_explorer.data_dict(king, king['price'])```
![Data_Dict Output](https://thumbs.dreamstime.com/z/tv-test-image-card-rainbow-multi-color-bars-geometric-signals-retro-hardware-s-minimal-pop-art-print-suitable-89603635.jpg)