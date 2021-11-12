#!/usr/bin/python
"""
Collection of support functions for EDA.
"""
# Libraries import
import time, warnings
import numpy as np
import pandas
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder
from itertools import product
from lib.ml_oversampling import MLSol

# Warnings turn off
warnings.simplefilter('ignore', np.RankWarning)
warnings.filterwarnings('ignore')

pd.set_option("display.max.columns", None)
plt.style.use('seaborn')

SMALL_SIZE = 12
MEDIUM_SIZE = 14
LARGE_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=LARGE_SIZE)   # fontsize of the figure title


def print_corr_matrix(df):
    """
    Function plots features correlation matrix from prepared dataframe.
    :param df: Df with features
    :return: Correlation matrix plot
    """
    # Clean and prepare df for correlation matrix plot
    df_cat = df.copy(deep=True)
    df_cat = df_cat.dropna()

    # Compute the correlation matrix
    corr = df_cat.corr()

    # Generate a mask for the upper triangle
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # Set up the matplotlib figure
    fig, ax = plt.subplots(figsize=(20, 18))

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)

    # Plot the matrix
    sns.heatmap(corr, mask=mask, annot=True, square=True, linewidths=.5, vmin=-1, vmax=1, cmap=cmap, ax=ax)
    plt.title("Amphibians Correlation matrix", fontsize=MEDIUM_SIZE, fontweight='bold')
    plt.xlabel("Features", fontweight='bold', fontsize=SMALL_SIZE, )
    plt.ylabel("Features", fontweight='bold', fontsize=SMALL_SIZE, )
    plt.tight_layout()
    plt.savefig('export/amphibians_corr_matrix.pdf', dpi=1000)
    plt.show()

    return corr

def remove_outliers(df, SR_lim, NR_lim):
    """
    Removes selected rows where SR and NR exceed the defined limits
    :param SR_lim: (int) limit
    :param NR_lim: (int) limit
    :return: Df with removed outliers
    """
    # Dropping outliers rows based on SR category limit
    to_drop_1 = df[df['SR'] > SR_lim]
    df_mod = df.drop(index=to_drop_1.index)
    df_mod.reset_index(inplace=True, drop=True)

    # Dropping outliers rows based on NR category limit
    to_drop_2 = df_mod[df_mod['NR'] > NR_lim]
    df_mod.drop(index=to_drop_2.index, inplace=True)
    df_mod.reset_index(inplace=True, drop=True)

    return df_mod

def encode_labels(df):
    """
    Encode features containing categorical labels
    :param df: Df with features containing categorical labels
    :return: Encoded Df
    """
    X_cat_enc = pd.DataFrame()
    for col in df.columns:
        array_enc = LabelEncoder().fit_transform(df[col])
        array_enc = array_enc.reshape(array_enc.shape[0], 1)
        df_enc = pd.DataFrame(data=array_enc, columns=[col])
        X_cat_enc = pd.concat([X_cat_enc, df_enc], axis=1, ignore_index=False)

    return X_cat_enc

def run_ml_sampling(X, y, ratios, neighbors):
    """
    Perform multi-label oversampling of the input dataset.
    :param X: Features DF
    :param y: Targets DF
    :param ratios: (array) Oversampling ratio
    :param neighbors: (array) Neighbors
    :return: Oversampled features and targets DF
    """
    # Instantiate sampler
    sampler = MLSol()

    # Generate more instances with nearest neighbors and search for the
    # best configuration
    deltas = []
    for r, n in product(ratios, neighbors):
        X_s, y_s = sampler.oversample(X.values, y.values, r, n)
        delta = abs(0.5 - np.mean(y_s))  # Calculate balance difference

        # Compile results
        result = pd.DataFrame(
            {'diff': delta,
             'ratio': r,
             'neighbor': n},
            index=[0])
        deltas.append(result)

    # Sort results for the smallest difference and locate relevant parameters
    results = pd.concat(deltas)
    results.sort_values('diff', ascending=True, inplace=True)
    results.reset_index(drop=True, inplace=True)

    # Select the best setup
    best_ratio = results['ratio'].loc[0]
    best_neighbor = results['neighbor'].loc[0]

    # Generate balanced dataset
    X_s, y_s = sampler.oversample(X.values, y.values, best_ratio, best_neighbor)
    X_s = pd.DataFrame(data=X_s)
    y_s = pd.DataFrame(data=y_s)

    return X_s, y_s

def concat_df(main, pred, preffix):
    """
    Function adds predicted volumes to evaluation dataframe
    :param main: Df with evaluation data
    :param pred: Predicted volume
    :return: Concatenated Df
    """
    is_df = ()

    try:
        is_df = pred.values.min
    except: AttributeError('Not and DF type')


    if not is_df:
        y_cols = ['Green frogs', 'Brown frogs', 'Common toad', 'Fire-bellied toad',
                  'Tree frog', 'Common newt', 'Great crested newt']
        y_cols_new = []
        for col in y_cols:
            col_new = col + '-' + preffix
            y_cols_new.append(col_new)
        pred = pd.DataFrame(data=pred, index=main.index, columns=y_cols_new)

    con = pd.concat([main, pred], axis=1, ignore_index=False)

    return con


if __name__ == "__main__":
    # Runtime initiation
    run_start = time.time()
    print('Running functions...\n')

    # Evaluate runtime
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print('\nAll ran in:', '%d:%02d:%02d'%(h, m, s))