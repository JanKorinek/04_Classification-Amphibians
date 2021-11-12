#!/usr/bin/python
"""
Collection of functions for data ingestion and datasets modification.
"""
#Libraries Import
import time, warnings
import pandas as pd

# Warnings turn off
warnings.filterwarnings('ignore')
warnings.warn('ignore')

if __name__ == "__main__":
    # Runtime initiation
    run_start = time.time()
    print('Processing raw data...')

    # Raw data import
    raw_amphibians = pd.read_csv('data/dataset.csv', sep=';', header=1)

    # Drop ID and MV columns
    amphibians = raw_amphibians.drop(columns=['ID', 'Motorway'])

    # Save dataset
    amphibians.to_pickle('data/amphibians.pickle')

    # Evaluate runtime
    m, s = divmod(time.time()-run_start,60)
    h, m = divmod(m, 60)
    print('Dataset processing finished in:', '%d:%02d:%02d'%(h, m, s))
