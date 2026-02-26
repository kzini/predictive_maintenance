import pandas as pd

def load_data(file_path):
    columns = ['unit_number', 'time_in_cycles'] + \
              [f'op_setting_{i}' for i in range(1, 4)] + \
              [f'sensor_{i}' for i in range(1, 22)]
    
    df = pd.read_csv(file_path, sep=r'\s+', header=None, names=columns)

    return df

def get_df_info(df):
    df_info = pd.concat(
        [
            df.isna().sum(),
            df.nunique(),
            df.dtypes
        ],
        axis=1
    )

    df_info.columns = ['missing_values', 'n_unique', 'dtype']
    return df_info

def constant_features(df, threshold=0.02):
    return [
        col for col in df.columns
        if col.startswith('sensor_') and df[col].std() < threshold
    ]

def calculate_rul(df):
    max_cycles = df.groupby('unit_number')['time_in_cycles'].transform('max')
    df['RUL'] = max_cycles - df['time_in_cycles']
    return df

