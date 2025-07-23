import warnings
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
import os
warnings.filterwarnings("ignore", category=UserWarning)

class LSTMWithFuture(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=0.15, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_seq):
        _, (hn, _) = self.lstm(x_seq)
        return self.fc(hn[-1]).squeeze(-1)
    
class CNNLSTMWithFuture(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, cnn_out_channels=32, kernel_size=3):
        super().__init__()
        
        self.conv1d = nn.Conv1d(
            in_channels=input_size, 
            out_channels=cnn_out_channels, 
            kernel_size=kernel_size, 
            padding=kernel_size // 2
        )
        self.relu = nn.ReLU()
        
        self.lstm = nn.LSTM(
            input_size=cnn_out_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=0.15,
            batch_first=True
        )
        
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_seq):  
        x_seq = x_seq.permute(0, 2, 1) 
        x_seq = self.relu(self.conv1d(x_seq))  
        x_seq = x_seq.permute(0, 2, 1) 
        
        # LSTM + output
        _, (hn, _) = self.lstm(x_seq)
        return self.fc(hn[-1]).squeeze(-1)

def predict_Tin_sequence_full(
    df_test,               
    window_start_idx,      
    lstm_model,               
    rf_model,   
    tin_scaler,
    rh_model,
    rh_scaler,
    window_size=6,  
    window_size_rh=24,     
    n_steps=96,
    hvac_schedule=None,
    debug=False
):
    lstm_model.eval()
    rh_model.eval()

    results = []

    tin_input_seq = []
    rh_input_seq = []

    for i in range(window_size):
        row_tin = df_test.iloc[window_start_idx - window_size + i]
        tin_input_row = [
            row_tin['Tin'], row_tin['Tout'],
            # row['Tin_lag1'], row['Tin_lag2'], row['Tin_lag3'], row['Tin_lag4'],
            row_tin['T_diff']
        ]          
        
        tin_scaled = tin_scaler.transform([tin_input_row])[0]
        tin_scaled_final = list(tin_scaled)
        tin_input_seq.append(tin_scaled_final)

    for i in range(window_size_rh):
        row_rh = df_test.iloc[window_start_idx - window_size_rh + i]
        rh_raw_features = [
            row_rh['RH'], row_rh['Tout']
        ]
        hvac_0 = row_rh['hvac_0']
        hvac_1 = row_rh['hvac_1']
        hvac_2 = row_rh['hvac_2']
        rh_scaled = rh_scaler.transform([rh_raw_features])[0]
        rh_input_row = list(rh_scaled) + [hvac_0, hvac_1, hvac_2] + [row_rh['hour_sin'], row_rh['hour_cos']]
        rh_input_seq.append(rh_input_row)

    start_row = df_test.iloc[window_start_idx]
    pred_tin_buffer = [
        start_row['Tin']
    ]

    pred_rh_buffer = [
        start_row['RH']
    ]

    results = []

    for step in range(n_steps):
        idx = window_start_idx + step
        row = df_test.iloc[idx]
        Tout = row['Tout']
        hvac_mode = row['hvac_mode']
        hvac_0 = row['hvac_0']
        hvac_1 = row['hvac_1'] 
        hvac_2 = row['hvac_2']
        if hvac_schedule is not None:
            hvac_0 = 1 if hvac_schedule[step] == 0 else 0
            hvac_1 = 1 if hvac_schedule[step] == 1 else 0
            hvac_2 = 1 if hvac_schedule[step] == 2 else 0
            hvac_mode = hvac_schedule[step]
        else:
            hvac_mode = row['hvac_mode']
            hvac_0 = row['hvac_0']
            hvac_1 = row['hvac_1'] 
            hvac_2 = row['hvac_2']
        if debug:
            print(f"\n======= STEP {step} =======")
            print(f"Timestamp: {df_test.index[idx]}")
            print(f"HVAC mode: {hvac_mode}")


        if hvac_mode == 0:

            tin_input = [
                pred_tin_buffer[0],       
                Tout,          
                row['T_diff'],    
            ]

            scaled_input_tin = tin_scaler.transform([tin_input])[0]

            final_scaled_tin = list(scaled_input_tin)
            tin_input_seq = tin_input_seq[1:] + [final_scaled_tin]
            x_seq = torch.tensor([tin_input_seq], dtype=torch.float32)
  
           
            with torch.no_grad():
                Tin_pred = lstm_model(x_seq).item()

         
        else:
            rf_input = pred_tin_buffer.copy() + [hvac_mode] + [Tout] + [row['T_diff']]
            Tin_pred = rf_model.predict([rf_input])[0]
     
     
        pred_tin_buffer = [Tin_pred] 

        rh_raw_features =  [pred_rh_buffer[0], Tout]
        scaled_rh_features = rh_scaler.transform([rh_raw_features])[0]
        rh_input_row = list(scaled_rh_features) + [hvac_0, hvac_1, hvac_2, row['hour_sin'], row['hour_cos']]
        rh_input_seq = rh_input_seq[1:] + [rh_input_row]
        rh_seq_tensor = torch.tensor([rh_input_seq], dtype=torch.float32)
   
        with torch.no_grad():
            RH_pred = rh_model(rh_seq_tensor).item()

        pred_rh_buffer = [RH_pred]
    
        results.append({
            'timestamp': df_test.index[idx],
            'Tin_pred': Tin_pred,
            'Tin_true': row['Tin'],
            'hvac_mode': hvac_mode,
            'Tin_target': row['Tin_target'],
            'RH_pred': RH_pred,
            'RH_true': row['RH'],
        })

        # lag_buffer = [Tin_pred] + lag_buffer[:-1]

    return pd.DataFrame(results).set_index('timestamp')


def predict_next_step(
    df_test,
    current_idx,        
    schedule,             
    lstm_model,
    rf_model,
    tin_scaler,
    rh_model,
    rh_scaler,
    window_size=6,
    window_size_rh=24
):
    lstm_model.eval()
    rh_model.eval()

    # Prepare Tin input sequence
    tin_input_seq = []
    for i in range(window_size):
        row = df_test.iloc[current_idx - window_size + i]
        features = [row['Tin'], row['Tout'], row['T_diff']]
        scaled = tin_scaler.transform([features])[0]
        tin_input_seq.append(list(scaled))

    # Prepare RH input sequence
    rh_input_seq = []
    for i in range(window_size_rh):
        row = df_test.iloc[current_idx - window_size_rh + i]
        rh_raw = [row['RH'], row['Tout']]
        scaled = rh_scaler.transform([rh_raw])[0]

        # Determine HVAC mode (from schedule if available, else fallback)
        t = i - (window_size_rh - len(schedule))
        hvac_mode = schedule[t] if 0 <= t < len(schedule) else row['hvac_mode']
        hvac_0 = 1 if hvac_mode == 0 else 0
        hvac_1 = 1 if hvac_mode == 1 else 0
        hvac_2 = 1 if hvac_mode == 2 else 0

        rh_input = list(scaled) + [hvac_0, hvac_1, hvac_2, row['hour_sin'], row['hour_cos']]
        rh_input_seq.append(rh_input)

    # For the prediction step (current_idx + 1)
    next_row = df_test.iloc[current_idx + 1]
    Tout = next_row['Tout']
    T_diff = next_row['T_diff']

    # HVAC mode for next step
    if len(schedule) >= 1:
        next_hvac = schedule[-1]
    else:
        next_hvac = next_row['hvac_mode']

    if next_hvac == 0:
        # Use LSTM
        tin_input = [df_test.iloc[current_idx]['Tin'], Tout, T_diff]
        scaled = tin_scaler.transform([tin_input])[0]
        tin_input_seq = tin_input_seq[1:] + [list(scaled)]
        x_seq = torch.tensor([tin_input_seq], dtype=torch.float32)
        with torch.no_grad():
            Tin_pred = lstm_model(x_seq).item()
    else:
        # Use RF
        rf_input = [df_test.iloc[current_idx]['Tin']] + [next_hvac, Tout, T_diff]
        Tin_pred = rf_model.predict([rf_input])[0]

    # Predict RH
    rh_raw = [df_test.iloc[current_idx]['RH'], Tout]
    scaled_rh = rh_scaler.transform([rh_raw])[0]
    hvac_0 = 1 if next_hvac == 0 else 0
    hvac_1 = 1 if next_hvac == 1 else 0
    hvac_2 = 1 if next_hvac == 2 else 0
    rh_input = list(scaled_rh) + [hvac_0, hvac_1, hvac_2, next_row['hour_sin'], next_row['hour_cos']]
    rh_input_seq = rh_input_seq[1:] + [rh_input]
    rh_seq_tensor = torch.tensor([rh_input_seq], dtype=torch.float32)
    with torch.no_grad():
        RH_pred = rh_model(rh_seq_tensor).item()

    return Tin_pred, RH_pred