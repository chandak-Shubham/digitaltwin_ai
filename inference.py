import json
import os
import torch
import pandas as pd
from model import LSTMModel
from preprocessing import load_preprocessor, FEATURE_COLS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InferencePipeline:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.preprocessor = load_preprocessor(os.path.join(models_dir, 'preprocessor.pkl'))
        
        with open(os.path.join(models_dir, 'model_config.json'), 'r') as f:
            config = json.load(f)
            
        self.model = LSTMModel(config['inputs'], config['hidden'], config['layers'], config['dropout']).to(DEVICE)
        self.model.load_state_dict(torch.load(os.path.join(models_dir, 'station_lstm_model.pth'), map_location=DEVICE, weights_only=True))
        self.model.eval()
        
        # Get feature names from preprocessor
        num_features = self.preprocessor.transformers_[0][2]
        cat_features = self.preprocessor.transformers_[1][1].named_steps['onehot'].get_feature_names_out(self.preprocessor.transformers_[1][2])
        self.feature_names = list(num_features) + list(cat_features)
        
    def predict_sequence(self, df_sequence):
        # Preprocess
        X_matrix = self.preprocessor.transform(df_sequence[FEATURE_COLS]).astype('float32')
        X_tensor = torch.tensor(X_matrix).unsqueeze(0).to(DEVICE) # shape: [1, seq_len, num_features]
        
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = torch.sigmoid(logits).cpu().numpy()[0]
            
        return probs, X_tensor
