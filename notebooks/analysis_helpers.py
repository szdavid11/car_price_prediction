import pandas as pd
import numpy as np


def get_correlations(df):
    # Check the correlation among numeric features
    numeric_features = df.select_dtypes(include=["float", "int"]).columns
    df_corr = df[numeric_features].corr()
    df_corr = df_corr.reset_index(names="variable1")

    df_melt_corr = df_corr.melt(
        id_vars="variable1", var_name="variable2", value_name="correlation"
    )
    df_melt_corr = df_melt_corr[
        df_melt_corr.variable1 != df_melt_corr.variable2
    ]  # Remove self correlations

    # Remove duplications
    df_melt_corr[["variable1", "variable2"]] = np.sort(
        df_melt_corr[["variable1", "variable2"]].values, axis=1
    )
    df_melt_corr.drop_duplicates(inplace=True)

    df_melt_corr.sort_values("correlation", ascending=False, inplace=True)
    return df_melt_corr


# There are clear outliers in the targer varibles that are much higher than the usual
# To handle this I simple remove this values from the dataset. Since I not even consider to buy such an expensive car.
def cap_values(df, column, multiplier=3):
    # Calculate the IQR of the column
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    # Define the upper whisker
    upper_whisker = Q3 + multiplier * IQR

    # Replace values above the upper whisker with the value of the upper whisker
    df[column] = df[column].apply(lambda x: upper_whisker if x > upper_whisker else x)

    return df


# Some feature has extramily hihg values compare
def has_outliers(df, col, multiplier=3):
    """
    This method uses the IQR scores to eliminate outliers.
    IQR is the range between the first quartile (25th percentile) and the third quartile (75th percentile).
    In the IQR method, a data point is considered as an outlier
    if it is below the first quartile – 1.5IQR or above the third quartile + 1.5IQR.
    """
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - multiplier * IQR
    upper_bound = Q3 + multiplier * IQR

    return any((df[col] < lower_bound) | (df[col] > upper_bound))


def bucketing(df, column, threshold):
    # bin/bucket these low-frequency categories into a new category, like 'Other'.
    value_counts = df[column].value_counts()
    rare_cats = value_counts[value_counts < threshold].index
    if len(rare_cats) > 0:
        df[column] = df[column].replace(rare_cats, "Other")

        # Combine 'Other' with the next smallest category if 'Other' is still rare
        while df[column].value_counts().loc["Other"] < threshold:
            # find the smallest category that is not 'Other'
            small_cat = (
                df[column]
                .value_counts()[df[column].value_counts().index != "Other"]
                .idxmin()
            )
            # combine 'Other' and the smallest category
            df[column] = df[column].replace({small_cat: "Other"})

    # If only 1 uniqu value is left then drop the colum
    if len(df[column].unique()) == 1:
        df.drop(columns=[column], inplace=True)
        print("Drop column", column)

    return df
