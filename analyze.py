import numpy
import pandas
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
import os, sys

def analyze(dataframe):
    # Cluster rows
    # TODO KPrototypes requires continuous numerical values, checken-egg problem
    # TODO Detect column types for clustering, KPrototypes needs categorial column indexes
    clusterer = KPrototypes(n_clusters=5, init='Cao', n_init=5, verbose=1, n_jobs=-1)
    clusters = clusterer.fit_predict(dataframe, categorical=[6, 7])

    # Add cluster id as column to dataframe
    dataframe['Cluster'] = clusters
    print(clusterer.cluster_centroids_)
    print(dataframe)

if __name__== "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = os.path.join(current_path, 'data', 'in.csv')

    print('Analyzing', filename, '...')

    data = pandas.read_csv(filename)
    analyze(data)

    print('Done')
