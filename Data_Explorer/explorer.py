#!/usr/bin/env python

import pandas as pd
import numpy as np


class data:

    def __init__(self, df, target):
        '''Create an instance of the data class
        Keyword arguments:
        df -- a pandas DataFrame to be summarized
        target -- a Pandas Series containing the target that corresponds to the
                  DataFrame
        '''
        self.df = df
        self.target = target

    def get_cardinality(self, include_numeric=True):
        '''Determine cardinality of each column.

        Keyword arguments:
        include_numeric -- If true, determine cardinality of all columns. If
        false, determine only for non-numeric columns

        Returns:
        A Pandas Series, containing each of the column names as the index
        values and a count of the unique values within each column as the data.
        '''
        if include_numeric:
            cols = self.df.columns
        else:
            cols = self.df.select_dtypes(exclude='number').columns
        uniques = []
        for col in cols:
            uniques.append(self.df[col].nunique())
        uniques = pd.Series(uniques, cols)
        return uniques

    def corr_with_target(self):
        '''Determine correlations of numeric columns with a specified target.

        Correlation values will not be returned for non-
        numeric datatypes. The length of the target Series must equal the
        dataframe.

        Returns:
        A Pandas Series containing the correlations of each feature with
        the specific target. Correlation values will not be returned for non-
        numeric datatypes.

        If target is non-numeric, it will be ordinal encoded. This may imply an
        ordering to the target classes that is improper.
        '''
        if (self.target.dtype != ('int' or 'float')):
            # TODO ordinal encode target
            target = pd.Series(self.target.factorize()[0])
            target = target.rename('target')
            frame = pd.concat([self.df, target], axis=1)
        else:
            # Concat features and target
            self.target = self.target.rename('target')
            frame = pd.concat([self.df, self.target], axis=1)
        corrs = frame.corr()

        # Drop target's correlation with itself
        corrs_target = corrs['target'].sort_values(ascending=False)[1:]

        return corrs_target

    def detect_suspicious(self, custom_vals=[]):
        '''Searches a DataFrame for values that should possibly be considered np.nan
        in a given DataFrame

        Keyword arguments:
        custom_vals -- A list containing additional unique values to be
        searched for (in addition to the built-in suspicious values)

        Returns:
        A Pandas DataFrame with an index listing the columns containing
        suspicious values, a column containing the number of suspicious values
        in each column, and a third column with a list of the suspicious
        values in each column
        '''
        suspicious_vals = ['Not Known', 'Unknown', 'None', 'Not known',
                           'not known', '-', 'unknown', '##', 'none',
                           '?', '??', '0', -9, -99, -999, -9999, 9999,
                           66, 77, 88, "NA", "N A", "N/A", "NA ", " NA",
                           "N /A", "N / A", " N / A", "N / A ", "na",
                           "n a", "n/a", "na ", " na", "n /a",
                           "n / a", " a / a", "n / a ", "NULL",
                           "null", "", "\\?", "\\*", "\\."]
        suspicious_vals = suspicious_vals + custom_vals
        cols = self.df.columns
        suspect_cols = []
        suspect_counts = []
        suspect_vals = []
        for col in cols:
            suspect = (self.df[col].isin(suspicious_vals)).sum()
            if (suspect > 0):
                suspect_cols.append(col)
                suspect_counts.append(suspect)

                # Find intersection of values in suspicious_vals and the column
                vals = list(set(self.df[col]) & set(suspicious_vals))
                suspect_vals.append(vals)
        frame = pd.DataFrame({
            'Suspicious Count': suspect_counts,
            'Suspect Values': suspect_vals
        }, index=suspect_cols)
        return frame

    def outlier_modified_z_score(self, threshold):
        '''Identify outliers in numeric columns using the modified z score range
        method. A modified z score uses the median rather than the mean.

        Keyword arguments:
        threshold -- a scalar, higher values decrease the number of outliers
                    that will be found

        Returns:
        A Pandas DataFrame with an index listing the columns containing
        outliers, a column containing the number of outliers in each column,
        and a third column with a list of the outliers in each column
        '''
        num_cols = self.df.select_dtypes(include='number')
        outlier_cols = []
        outlier_counts = []
        outlier_vals = []
        for col in num_cols.columns:
            outliers = []
            median = np.median(self.df[col])
            median_absolute_deviation = np.median([np.abs(i - median) for i in
                                                  self.df[col]])

            modified_z_scores = []
            for i in self.df[col]:
                if median_absolute_deviation != 0:
                    modified_z_scores.append(0.675 * (i-median) /
                                             median_absolute_deviation)
                else:
                    modified_z_scores.append(0)

            outliers = np.where(np.abs(modified_z_scores) > threshold)

            outliers_set = set()
            for i in outliers[0]:
                outlier_val = self.df[col].iloc[i]
                outliers_set.add(outlier_val)

            if outliers[0].size > 0:
                outlier_cols.append(col)
                outlier_counts.append(len(outliers[0]))
                outlier_vals.append(outliers_set)

        frame = pd.DataFrame({
                'Outlier Count': outlier_counts,
                'Outlier Values': outlier_vals
        }, index=outlier_cols)

        return frame

    def outlier_z_score(self, threshold):
        '''Identify outliers in numeric columns using the z-score method.

        Keyword arguments:
        threshold -- a scalar, higher values decrease the number of outliers
                    that will be found

        Returns:
        A Pandas DataFrame with an index listing the columns containing
        outliers, a column containing the number of outliers in each column,
        and a third column with a list of the outliers in each column
        '''
        num_cols = self.df.select_dtypes(include='number')
        outlier_cols = []
        outlier_counts = []
        outlier_vals = []
        for col in num_cols.columns:
            outliers = []
            mean = np.mean(self.df[col])
            stdev = np.std(self.df[col])
            z_scores = [(i - mean) / stdev for i in self.df[col]]

            outliers = np.where(np.abs(z_scores) > threshold)

            outliers_set = set()
            for i in outliers[0]:
                outlier_val = self.df[col].iloc[i]
                outliers_set.add(outlier_val)

            if outliers[0].size > 0:
                outlier_cols.append(col)
                outlier_counts.append(len(outliers[0]))
                outlier_vals.append(outliers_set)

        frame = pd.DataFrame({
                'Outlier Count': outlier_counts,
                'Outlier Values': outlier_vals
            }, index=outlier_cols)

        return frame

    def outlier_iqr(self, threshold):
        '''Identify outliers in numeric columns using the interquartile range method.

        Keyword arguments:
        threshold -- a scalar, higher values decrease the number of outliers
                    that will be found

        Returns:
        A Pandas DataFrame with an index listing the columns containing
        outliers, a column containing the number of outliers in each column,
        and a third column with a list of the outliers in each column
        '''
        num_cols = self.df.select_dtypes(include='number')
        outlier_cols = []
        outlier_counts = []
        outlier_vals = []
        for col in num_cols.columns:
            outliers = []
            quartile_1, quartile_3 = np.percentile(self.df[col], [25, 75])
            iqr = quartile_3 - quartile_1
            lower = quartile_1 - (iqr * threshold)
            upper = quartile_3 + (iqr * threshold)

            outliers = np.where((self.df[col] > upper) | (self.df[col] < lower)
                                )

            outliers_set = set()
            for i in outliers[0]:
                outlier_val = self.df[col].iloc[i]
                outliers_set.add(outlier_val)

            if outliers[0].size > 0:
                outlier_cols.append(col)
                outlier_counts.append(len(outliers[0]))
                outlier_vals.append(outliers_set)

        frame = pd.DataFrame({
                'Outlier Count': outlier_counts,
                'Outlier Values': outlier_vals
            }, index=outlier_cols)

        return frame

    def detect_outliers(self, method='modified_z_score', threshold=3.5):
        '''Identify outliers in numeric columns using the specified outlier
        detection method.

        Keyword arguments:
        threshold -- a scalar, higher values decrease the number of outliers
            that will be found

        Returns:
        A Pandas DataFrame with an index listing the columns containing
        outliers, a column containing the number of outliers in each
        column, and a third column with a list of the outliers in each column
        '''
        if method == 'standard_z_score':
            result = self.outlier_standard_z_score(threshold)
        elif method == 'outlier_iqr':
            result = self.outlier_iqr(threshold)
        else:
            result = self.outlier_modified_z_score(threshold)
        return result

    def data_dict(self):
        '''Create a DataFrame intended to orient the user to a dataset.

        Returns:
        A Pandas DataFrame
        '''
        null = self.df.isnull().sum()
        null = pd.DataFrame(null)
        null.columns = ['Null Values']

        unique = self.df.nunique()
        unique = pd.DataFrame(unique)
        unique.columns = ['Unique Values']

        cardinality = self.get_cardinality()
        cardinality = pd.DataFrame(cardinality)
        cardinality.columns = ['Cardinality']

        datatype = self.df.dtypes
        datatype = pd.DataFrame(datatype)
        datatype.columns = ['Datatype']

        skew = self.df.skew()
        skew = pd.DataFrame(skew)
        skew.columns = ['Skew']

        corr_w_target = self.corr_with_target()
        corr_w_target = pd.DataFrame(corr_w_target)
        corr_w_target.columns = ['Correlation W/Target']

        suspicious = self.detect_suspicious()
        outlier = self.detect_outliers()

        frames = [null, cardinality, datatype, skew, suspicious, corr_w_target,
                  outlier]

        combined = unique
        for frame in frames:
            combined = pd.merge(combined, frame, how='outer', left_index=True,
                                right_index=True)

        counts = ['Suspicious Count', 'Outlier Count']
        combined[counts] = combined[counts].fillna(0)
        vals = ['Suspect Values', 'Outlier Values']
        combined[vals] = combined[vals].fillna('None')
        non_num = ['Skew', 'Correlation W/Target']
        combined[non_num] = combined[non_num].fillna('Non-Numeric')

        col_ordered = ['Datatype', 'Unique Values', 'Null Values',
                       'Cardinality', 'Correlation W/Target', 'Skew',
                       'Outlier Count', 'Outlier Values',
                       'Suspicious Count', 'Suspect Values']
        combined = combined[col_ordered]

        return combined
