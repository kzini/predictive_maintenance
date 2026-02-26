import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, cross_val_predict

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f'RMSE = {rmse:0.2f}, R² = {r2*100:0.2f}%')

def train_test_split_by_group(df, group_col='unit_number', test_size=0.2, random_state=42):
    from sklearn.model_selection import GroupShuffleSplit
    
    gss = GroupShuffleSplit(
        test_size=test_size, 
        n_splits=1, 
        random_state=random_state
    )
    
    train_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_test = df.iloc[test_idx].reset_index(drop=True)
    
    return df_train, df_test

def evaluate_regression(model, X, y, groups=None, cv_splits=5):
    pipeline = make_pipeline(model)
    
    if groups is not None:
        cv = GroupKFold(n_splits=cv_splits)
        y_pred = cross_val_predict(pipeline, X, y, groups=groups, cv=cv, n_jobs=-1)
    else:
        y_pred = cross_val_predict(pipeline, X, y, cv=cv_splits, n_jobs=-1)
    
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)
    
    return {'RMSE': rmse, 'R2': r2*100, 'y_pred': y_pred}

def train_and_evaluate_model(X_train, X_test, y_train, y_test, model, model_name):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    metrics_df = pd.DataFrame(
        [[model_name, round(rmse, 2), round(r2*100, 2)]],
        columns=['Modelo', 'RMSE', 'R2 (%)']
    )
    
    return metrics_df, y_pred, model
