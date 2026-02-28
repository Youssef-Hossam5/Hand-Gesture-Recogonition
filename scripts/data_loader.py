import pandas as pd

def load_data(data_path):
    df = pd.read_csv(data_path)
    df.info()
    return df