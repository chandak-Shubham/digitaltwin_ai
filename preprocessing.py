import joblib
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

NUMERIC_COLS = ['cycle_time_sec', 'torque_nm', 'temperature_c', 'vibration_rms', 'pressure_bar', 'force_n', 'position_error_mm', 'voltage_v', 'current_a', 'flow_rate_lpm', 'queue_time_sec', 'ambient_temperature_c', 'humidity_pct']
CATEGORICAL_COLS = ['vehicle_model', 'vehicle_variant', 'station_id', 'shift', 'production_batch']
FEATURE_COLS = NUMERIC_COLS + CATEGORICAL_COLS

def create_preprocessor():
    prep = ColumnTransformer([
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='median')), 
            ('scale', StandardScaler())
        ]), NUMERIC_COLS), 
        ('cat', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')), 
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), CATEGORICAL_COLS)
    ])
    return prep

def save_preprocessor(prep, path):
    joblib.dump(prep, path)

def load_preprocessor(path):
    return joblib.load(path)
