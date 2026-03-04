import numpy as np
import pandas as pd

def add_degradation_features(
    df,
    sensor_cols,
    unit_col='unit_number',
    ma_windows=(10,),
    diff_windows = (1, 25, 90)
):
    df = df.copy()

    for s in sensor_cols:
        grouped = df.groupby(unit_col)[s]

        # Médias móveis
        for w in ma_windows:
            df[f'{s}_ma_{w}'] = (
                grouped
                .rolling(window=w, min_periods=1)
                .mean()
                .reset_index(level=0, drop=True)
            )

        # Diferença padrão (1 ciclo)
        df[f'{s}_diff'] = grouped.diff().fillna(0)

        # Diferenças em múltiplas escalas
        for w in diff_windows:
            if w == 1:
                continue
            df[f'{s}_diff_{w}'] = grouped.diff(periods=w).fillna(0)

        # Variação relativa percentual
        df[f'{s}_pct_change'] = (
            grouped
            .pct_change()
            .replace([np.inf, -np.inf], 0)
            .fillna(0)
        )

    return df

def calculate_health_index(df, sensor_list, healthy_cycles=25):
    health_series = pd.Series(index=df.index, dtype='float64')

    for unit in df['unit_number'].unique():
        motor_mask = df['unit_number'] == unit
        motor_data = df.loc[motor_mask, sensor_list]

        healthy_idx = min(healthy_cycles, len(motor_data))
        healthy_data = motor_data.iloc[:healthy_idx]

        healthy_mean = healthy_data.mean()
        healthy_std = healthy_data.std()

        safe_divisor = healthy_std.copy()

        low_std_mask = safe_divisor < 1e-6
        safe_divisor[low_std_mask] = 1e-6

        normalized_diff = (motor_data - healthy_mean) / safe_divisor

        health_index = np.sqrt((normalized_diff ** 2).sum(axis=1))

        health_series.loc[motor_mask] = health_index.values

    return health_series

