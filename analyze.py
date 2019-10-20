import numpy
import pandas
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from kmodes.kprototypes import KPrototypes
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import os, sys
import random
import copy

def analyze(dataframe):
    # Cluster rows
    # TODO KPrototypes requires continuous numerical values, chicken-egg problem
    # TODO Detect column types for clustering, KPrototypes needs categorial column indexes

    clusterer = KMeans(n_clusters=3, n_init=5, verbose=1, n_jobs=-1)
    clusters = clusterer.fit_predict(dataframe)

    #clusterer = KModes(n_clusters=3, init='Huang', n_init=5, verbose=1, n_jobs=-1)
    #clusters = clusterer.fit_predict(dataframe)

    #clusterer = KPrototypes(n_clusters=3, init='Cao', n_init=5, verbose=1, n_jobs=-1)
    #clusters = clusterer.fit_predict(dataframe, categorical=[6, 7])

    # Add cluster id as column to dataframe
    dataframe['Cluster'] = clusters

    # Train models of all cluster and columns
    models = {}
    for cluster_index in list(set(clusterer.labels_)):
        print()
        print('CLuster:', cluster_index)

        models[cluster_index] = {}
        cluster_data = dataframe[dataframe['Cluster'] == cluster_index]
        print(cluster_data.shape)

        for column_name in cluster_data.columns.values.tolist():
            if column_name == 'Cluster':
                continue

            # Collect data
            train_x = []
            train_y = []
            for row in cluster_data.itertuples():
                training_row = []

                # Add to training data as input or target
                for field_tuple in row._asdict().items():
                    field_name = field_tuple[0]
                    field_value = field_tuple[1]

                    if field_name == 'Index' or field_name == 'Cluster':
                        continue

                    if field_name == column_name:
                        # Skip row if target column data doesn't exist
                        if field_value == numpy.nan or field_value is None or field_value == '':
                            break

                        train_y.append(field_value)
                        continue
                    training_row.append(field_value)
                train_x.append(training_row)

            # Train model
            print('Training model for cluster', cluster_index, 'column', column_name)
            if(cluster_data[column_name].dtype == numpy.float64 or cluster_data[column_name].dtype == numpy.int64):
                # Treat as numeric
                model = RandomForestRegressor(n_estimators=100, max_depth=3)
            else:
                # Treat as categorical
                model = RandomForestClassifier(n_estimators=100, max_depth=3)

            model.fit(train_x, train_y)

            # Insert trained model to model set
            models[cluster_index][column_name] = model

    # TODO Save models

    return clusterer, models

def fill(dataframe, models, clusterer):
    dataset_fields = data.columns.values.tolist()

    for row in dataframe.sample(n=100).itertuples():
        # Add to prediction input data
        predict_row = []
        for field_tuple in row._asdict().items():
            field_name = field_tuple[0]
            field_value = field_tuple[1]

            if field_name == 'Index' or field_name == 'Cluster':
                continue

            predict_row.append(field_value)

        # Predict cluster of row
        cluster_index = clusterer.predict([predict_row])[0]

        # Select field at random and predict the value if it was missing
        selected_field = random.choice(dataset_fields)
        selected_field_index = dataset_fields.index(selected_field)

        # Drop "missing" value from prediction input data
        value_input_row = copy.copy(predict_row)
        del value_input_row[selected_field_index]
        prediction = models[cluster_index][selected_field].predict(
            [value_input_row]
        )

        print(
            'Field:', selected_field,
            'Predicted:', prediction[0],
            'Actual:', predict_row[selected_field_index],
            'Difference:', (predict_row[selected_field_index] - prediction[0])
        )

    return True

if __name__== "__main__":
    current_path = os.path.dirname(os.path.abspath(__file__))
    try:
        filename = sys.argv[1]
    except IndexError:
        filename = os.path.join(current_path, 'data', 'in.csv')

    print('Analyzing', filename, '...')

    data = pandas.read_csv(filename)
    clusterer, models = analyze(data)

    data = pandas.read_csv(filename)
    fill(data, models=models, clusterer=clusterer)

    print('Done')
