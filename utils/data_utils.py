import pandas as pd


def preprocess_dict_input(data, target_variable):
    df = pd.DataFrame(data)
    y = df[target_variable]
    X = df.drop(target_variable, axis=1)
    return X, y
