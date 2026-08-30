import os
import json
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from model import LSTMModel
from preprocessing import create_preprocessor, save_preprocessor, NUMERIC_COLS, CATEGORICAL_COLS, FEATURE_COLS

DATASET_PATH = 'dataset_variant_propagation.csv'
MODELS_DIR = 'models'
TARGET = 'anomaly_flag'
VEHICLE_ID = 'vehicle_id'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
SEED = 42

# Hyperparameters
BATCH_SIZE = 32
EPOCHS = 8
HIDDEN = 128
LAYERS = 2
DROPOUT = 0.2
LR = 0.001

def load_dataset(path):
    df = pd.read_csv(path).copy()
    for col in NUMERIC_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    return df

def split_vehicles(df):
    labels = df.groupby(VEHICLE_ID)[TARGET].max().astype(int)
    train_ids, test_ids = train_test_split(labels.index, test_size=0.2, random_state=SEED, stratify=labels)
    return df[df[VEHICLE_ID].isin(train_ids)].copy(), df[df[VEHICLE_ID].isin(test_ids)].copy()

def get_sequences(df, matrix, station_order):
    # Create mapping from (vehicle_id, station_id) to matrix row index
    df = df.copy()
    df['matrix_idx'] = np.arange(len(df))
    
    # We will build X and y manually to ensure exact station_order is respected
    X_list = []
    y_list = []
    
    # Group by vehicle
    groups = df.groupby(VEHICLE_ID)
    
    for v_id, group in groups:
        group_dict = group.set_index('station_id')
        
        # Verify if vehicle has all stations safely
        if len(group_dict) != len(station_order):
            print(f"Warning: Vehicle {v_id} does not have exactly {len(station_order)} stations. Padding missing.")
            
        v_x = []
        v_y = []
        for station in station_order:
            if station in group_dict.index:
                # Need to handle duplicate stations if they exist, take the first one
                if isinstance(group_dict.loc[station, 'matrix_idx'], pd.Series):
                    row_idx = group_dict.loc[station, 'matrix_idx'].iloc[0]
                    tgt = group_dict.loc[station, TARGET].iloc[0]
                else:
                    row_idx = group_dict.loc[station, 'matrix_idx']
                    tgt = group_dict.loc[station, TARGET]
                    
                v_x.append(matrix[row_idx])
                v_y.append(tgt.astype('float32'))
            else:
                # Handle missing safely: pad with zeros
                v_x.append(np.zeros(matrix.shape[1], dtype='float32'))
                v_y.append(0.0)
                
        X_list.append(np.stack(v_x))
        y_list.append(np.array(v_y, dtype='float32'))
        
    return np.stack(X_list), np.stack(y_list)

def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    print(f"Loading {DATASET_PATH}...")
    df = load_dataset(DATASET_PATH)

    print("Splitting dataset...")
    train_df, test_df = split_vehicles(df)

    # 1. Derive canonical station_order from the actual production flow using timestamps
    # Take a sample vehicle from train to find order
    sample_vehicle = train_df['vehicle_id'].iloc[0]
    station_order = train_df[train_df['vehicle_id'] == sample_vehicle].sort_values('timestamp')['station_id'].tolist()
    if len(station_order) == 0:
        station_order = sorted(train_df['station_id'].unique())
    
    with open(os.path.join(MODELS_DIR, 'station_order.json'), 'w') as f:
        json.dump(station_order, f, indent=4)

    # 2. Save station baselines using ONLY train_df to prevent data leakage
    print("Computing station baselines from training data...")
    station_baselines = {}
    for station in station_order:
        station_data = train_df[train_df['station_id'] == station]
        station_baselines[station] = {}
        for col in NUMERIC_COLS:
            val = station_data[col].median()
            station_baselines[station][col] = float(val) if not np.isnan(val) else 0.0
            
    with open(os.path.join(MODELS_DIR, 'station_baselines.json'), 'w') as f:
        json.dump(station_baselines, f, indent=4)
        
    # 3. Calculate categorical defaults using ONLY train_df
    print("Computing categorical defaults from training data...")
    categorical_defaults = {}
    for cat in ['vehicle_model', 'vehicle_variant', 'shift', 'production_batch']:
        categorical_defaults[cat] = str(train_df[cat].mode()[0])
        
    with open(os.path.join(MODELS_DIR, 'categorical_defaults.json'), 'w') as f:
        json.dump(categorical_defaults, f, indent=4)

    print("Preprocessing data...")
    prep = create_preprocessor()
    train_matrix = prep.fit_transform(train_df[FEATURE_COLS]).astype('float32')
    test_matrix = prep.transform(test_df[FEATURE_COLS]).astype('float32')
    
    save_preprocessor(prep, os.path.join(MODELS_DIR, 'preprocessor.pkl'))

    print("Generating sequences...")
    X_train, y_train = get_sequences(train_df, train_matrix, station_order)
    X_test, y_test = get_sequences(test_df, test_matrix, station_order)

    print(f"Training shapes: X={X_train.shape}, y={y_train.shape}")
    
    model = LSTMModel(X_train.shape[2], HIDDEN, LAYERS, DROPOUT).to(DEVICE)
    
    pos = max(float(y_train.sum()), 1)
    neg = max(float(y_train.size - y_train.sum()), 1)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / pos], device=DEVICE))
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    loader = DataLoader(TensorDataset(torch.tensor(X_train), torch.tensor(y_train)), batch_size=BATCH_SIZE, shuffle=True)
    
    print("Training model...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb.to(DEVICE)), yb.to(DEVICE))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {total_loss/len(loader):.4f}")
        
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'station_lstm_model.pth'))
    
    # Save model config to recreate it easily
    config = {
        'inputs': X_train.shape[2],
        'hidden': HIDDEN,
        'layers': LAYERS,
        'dropout': DROPOUT
    }
    with open(os.path.join(MODELS_DIR, 'model_config.json'), 'w') as f:
        json.dump(config, f, indent=4)
        
    print("Extracting representative healthy vehicle...")
    from sklearn.preprocessing import StandardScaler
    vehicle_anomalies = train_df.groupby(VEHICLE_ID)[TARGET].max()
    healthy_vehicles = vehicle_anomalies[vehicle_anomalies == 0].index
    healthy_df = train_df[train_df[VEHICLE_ID].isin(healthy_vehicles)].copy()
    
    median_profile = healthy_df.groupby('station_id')[NUMERIC_COLS].median()
    
    scaler = StandardScaler()
    scaled_numeric = scaler.fit_transform(healthy_df[NUMERIC_COLS])
    healthy_df_scaled = healthy_df.copy()
    healthy_df_scaled[NUMERIC_COLS] = scaled_numeric
    
    scaled_median_profile = pd.DataFrame(
        scaler.transform(median_profile),
        index=median_profile.index,
        columns=median_profile.columns
    )
    
    def compute_distance(group):
        dist = 0
        for station in station_order:
            if station in group['station_id'].values:
                row = group[group['station_id'] == station][NUMERIC_COLS].iloc[0]
                prof = scaled_median_profile.loc[station]
                dist += np.sum((row - prof)**2)
        return dist
        
    distances = healthy_df_scaled.groupby(VEHICLE_ID).apply(compute_distance)
    best_vehicle_id = distances.idxmin()
    print(f"Selected reference vehicle: {best_vehicle_id}")
    
    reference_vehicle_df = train_df[train_df[VEHICLE_ID] == best_vehicle_id].copy()
    reference_vehicle_df['station_cat'] = pd.Categorical(reference_vehicle_df['station_id'], categories=station_order, ordered=True)
    reference_vehicle_df = reference_vehicle_df.sort_values('station_cat').drop('station_cat', axis=1)
    
    reference_vehicle_df.to_csv(os.path.join(MODELS_DIR, 'reference_vehicle.csv'), index=False)
        
    print("Training complete and artifacts saved.")

if __name__ == '__main__':
    main()
