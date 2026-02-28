import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def normalize_hand_row(row):
    wrist_x, wrist_y = row['x1'], row['y1']
    mid_x, mid_y = row['x12'] - wrist_x, row['y12'] - wrist_y
    scale = np.sqrt(mid_x**2 + mid_y**2)
    if scale == 0:
        scale = 1e-6

    new_row = {}
    for i in range(1, 22):
        x = (row[f'x{i}'] - wrist_x) / scale
        y = (row[f'y{i}'] - wrist_y) / scale
        new_row[f'x{i}'] = x
        new_row[f'y{i}'] = y
        new_row[f'z{i}'] = row[f'z{i}']
    return pd.Series(new_row)

def preprocess(df):
    X_norm = df.apply(normalize_hand_row, axis=1)
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)
    return X_norm.values, y, le
