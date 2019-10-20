import numpy
import pandas
import random

# Based on https://scikit-learn.org/stable/auto_examples/preprocessing/plot_map_data_to_normal.html


def generate(size):
    rng = numpy.random

    # lognormal distribution
    X_lognormal = rng.lognormal(size=size)

    # chi-squared distribution
    df = 3
    X_chisq = rng.chisquare(df=df, size=size)

    # weibull distribution
    a = 50
    X_weibull = rng.weibull(a=a, size=size)

    # gaussian distribution
    loc = 100
    X_gaussian = rng.normal(loc=loc, size=size)

    # uniform distribution
    X_uniform = rng.uniform(low=0, high=1, size=size)

    # bimodal distribution
    loc_a, loc_b = 100, 105
    X_a, X_b = rng.normal(loc=loc_a, size=int(size/2)), rng.normal(loc=loc_b, size=int(size/2))
    X_bimodal = numpy.concatenate([X_a, X_b], axis=0)

    # Two choice uniform categorical
    two_choice = [
        random.choice(['apple', 'orange']) for x in range(size)
    ]

    # Three choice center weighted categorical
    three_choice = [
        random.choices(['rain', 'sunny', 'windy'], weights=[4, 7, 3])[0] for x in range(size)
    ]

    data = {
        'Lognormal': X_lognormal,
        'ChiSquared': X_chisq,
        'Weibull': X_weibull,
        'Gaussian': X_gaussian,
        'Uniform': X_uniform,
        'Bimodal': X_bimodal,
        #'TwoChoice': two_choice,
        #'ThreeChoice': three_choice
    }
    dataframe = pandas.DataFrame(data)

    return dataframe

if __name__== "__main__":
    size = 10000
    dataframe = generate(size)

    print(dataframe.to_csv(index=False))
