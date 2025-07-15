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

def predict_Tin_sequence_full(
    df_test,               
    window_start_idx,      
    lstm_model,               
    rf_model,   
    tin_scaler,
    rh_model,
    rh_scaler,
    window_size=6,       
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
        row = df_test.iloc[window_start_idx - window_size + i]
        tin_input_row = [
            row['Tin'], row['Tout'],
            # row['Tin_lag1'], row['Tin_lag2'], row['Tin_lag3'], row['Tin_lag4'],
            row['T_diff']
        ]
        rh_raw_features = [
            row['RH'], row['Tout']
        ]
        hvac_0 = row['hvac_0']
        hvac_1 = row['hvac_1']
        hvac_2 = row['hvac_2']
                
        rh_scaled = rh_scaler.transform([rh_raw_features])[0]
        rh_input_row = list(rh_scaled) + [hvac_0, hvac_1, hvac_2] + [row['hour_sin'], row['hour_cos']]
        
        tin_scaled = tin_scaler.transform([tin_input_row])[0]
        tin_scaled_final = list(tin_scaled)
        
        rh_input_seq.append(rh_input_row)
        tin_input_seq.append(tin_scaled_final)

    # window_df = df_test.iloc[window_start_idx - window_size:window_start_idx].copy()

    # assert len(window_df) == window_size
    # input_seq = window_df[tin_input_cols].values.tolist()
    # input_seq = scaler.transform(window_df[tin_input_cols].values).tolist()
    start_row = df_test.iloc[window_start_idx]
    pred_tin_buffer = [
        start_row['Tin']
    ]

    pred_rh_buffer = [
        start_row['RH']
    ]

    # rh_input_cols = ['RH', 'Tout', 'hvac_0', 'hvac_1', 'hvac_2', 'hour_sin', 'hour_cos']
    # rh_input_seq = rh_scaler.transform(window_df[rh_input_cols].values).tolist()    
    
    # idxs = np.arange(window_start_idx, window_start_idx + n_steps)
    results = []

    # lag_buffer = [
    #     df_test['Tin'].iloc[window_start_idx - 0], 
    #     df_test['Tin'].iloc[window_start_idx - 1],  
    #     df_test['Tin'].iloc[window_start_idx - 2], 
    #     df_test['Tin'].iloc[window_start_idx - 3],  
    #     df_test['Tin'].iloc[window_start_idx - 4],  
    # ]

    # pred_tin_buffer = lag_buffer.copy() 

    for step in range(n_steps):
        idx = window_start_idx + step
        row = df_test.iloc[idx]
        Tout = row['Tout']
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

        # X_feat = lag_buffer.copy()

        if hvac_mode == 0:
            tin_input = [
                pred_tin_buffer[0],       
                Tout,          
                # pred_tin_buffer[1],        
                # pred_tin_buffer[2],
                # pred_tin_buffer[3],
                # pred_tin_buffer[4],
                # row['Tout_fut1'], 
                row['T_diff'],    
            ]

            scaled_input_tin = tin_scaler.transform([tin_input])[0]
            # scaled_input_tin = np.clip(scaled_input_tin, -1.5, 1.5)

            # print("pred_tin_buffer:", pred_tin_buffer)
            # print("predicted_input_row (raw):", tin_input)

            # print("input row (raw) Tin:", tin_input)
            # print("input row (scaled) Tin:", scaled_input_tin)
            final_scaled_tin = list(scaled_input_tin)
            tin_input_seq = tin_input_seq[1:] + [final_scaled_tin]
            x_seq = torch.tensor([tin_input_seq], dtype=torch.float32)
            with torch.no_grad():
                Tin_pred = lstm_model(x_seq).item()
            if debug:
                print(f"Tin_target (true next Tin): {df_test['Tin_target'].iloc[idx]:.3f}")
                print("Tin LSTM INPUT (raw):", tin_input)
                print("Tin LSTM INPUT (scaled):", scaled_input_tin)
                print("Tin input sequence shape:", x_seq.shape)
                print(f"LSTM Tin_pred: {Tin_pred:.3f}")
            # Tin_mean = scaler.mean_[0]
            # Tin_std = np.sqrt(scaler.var_[0])
            # Tin_pred_scaled = (Tin_pred - Tin_mean) / Tin_std
            # pred_tin_buffer = [Tin_pred] + pred_tin_buffer[:-1]

        else:
            rf_input = pred_tin_buffer.copy() + [hvac_mode] + [Tout] + [row['T_diff']]
            Tin_pred = rf_model.predict([rf_input])[0]
            if debug:
                print(f"RF Tin_pred: {Tin_pred:.3f}")
                print("RF Tin input features:", rf_input)
            # Tin_pred = rf_model.predict([X_feat + [hvac_mode] + [Tout]])[0]
            # pred_tin_buffer = [Tin_pred] + pred_tin_buffer[:-1]

        pred_tin_buffer = [Tin_pred] 

        rh_raw_features =  [pred_rh_buffer[0], Tout]
        scaled_rh_features = rh_scaler.transform([rh_raw_features])[0]
        # scaled_rh_features = np.clip(scaled_rh_features, -1.5, 1.5)
        rh_input_row = list(scaled_rh_features) + [hvac_0, hvac_1, hvac_2, row['hour_sin'], row['hour_cos']]
        rh_input_seq = rh_input_seq[1:] + [rh_input_row]
        rh_seq_tensor = torch.tensor([rh_input_seq], dtype=torch.float32)
        with torch.no_grad():
            RH_pred = rh_model(rh_seq_tensor).item()

        pred_rh_buffer = [RH_pred]
        if debug:
            print("RH MODEL INPUT (raw):", rh_raw_features)
            print("RH MODEL INPUT (scaled):", scaled_rh_features)
            print("RH input sequence shape:", rh_seq_tensor.shape)
            print(f"[{row.name}] RH_pred (raw): {RH_pred}")
            print(f"Updated RH buffer: {pred_rh_buffer}")
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