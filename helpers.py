from pythermalcomfort.models import pmv_ppd_iso
import pandas as pd
import numpy as np
from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from cycler import cycler
import json 

def calculate_overall_comfort(list_tin, list_rh):
    ppd_values = []
    for tin, rh in zip(list_tin, list_rh):
        comfort = pmv_ppd_iso(
            tdb=tin,
            tr=tin,     
            vr=0.1,     
            rh=rh,
            met=1.1,    
            clo=1.0     
        )
        if comfort['ppd'] is None:
            print(tin)
            ppd_values.append(75)  
        else:
            ppd_values.append(comfort['ppd'])  
    return ppd_values  

COMFORT_COLS = ["PMV", "PPD", "Overall_Comfort"]

def calculate_overall_comfort_df(row):
    results = pmv_ppd_iso(
        tdb=row['Tin'],
        tr=row['Tin'],  
        vr=0.1,      
        rh=row['RH'],
        met=1.1,         
        clo=1.0,         
    )
    pmv = results['pmv']
    ppd = results['ppd']
    comfort_pct = 100 - ppd
    return pd.Series({'PMV': pmv, 'PPD': ppd, 'Overall_Comfort': comfort_pct}, index=COMFORT_COLS)

def _es_hPa_from_Tc(Tc: float) -> float:
    """Saturation vapor pressure in hPa at air temperature Tc (°C)."""
    return 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))

def RH_percent_from_AH_T(AH_gm3: float, Tc: float) -> float:
    """Relative humidity (%) from absolute humidity (g/m³) and temperature (°C)."""
    es = _es_hPa_from_Tc(Tc)
    # vapor pressure from AH
    e = max(0.0, AH_gm3) * (Tc + 273.15) / 216.7
    RH = 100.0 * (e / max(1e-6, es))
    return float(np.clip(RH, 0.0, 100.0))

def AH_gm3_from_T_RH(Tc: float, RH_percent: float) -> float:
    """
    Absolute humidity [g/m³] from temperature (°C) and RH (%).
    AH = 216.7 * e / (T+273.15), with e = RH/100 * es(T).
    """
    T = np.asarray(Tc, dtype=np.float32)
    RHf = np.clip(np.asarray(RH_percent, dtype=np.float32) / 100.0, 0.0, 1.0)  # <-- vectorized clamp
    es = _es_hPa_from_Tc(T)  # assumes this already handles vector inputs (it does in your file)
    e = RHf * es
    AH = 216.7 * e / (T + 273.15)
    if isinstance(Tc, pd.Series):
        return pd.Series(AH, index=Tc.index, name="AH")
    return AH

def make_splits_by_is_test_or_date(
    df: pd.DataFrame,
    val_start_str: str = "2023-01-22",
    val_days: int = 14,
    test_days: int = 14,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    if not isinstance(df.index, pd.DatetimeIndex):
        if "timestamps" in df.columns:
            df = df.copy()
            df["timestamps"] = pd.to_datetime(df["timestamps"])
            df = df.set_index("timestamps")
        else:
            raise ValueError("DataFrame must have a DateTimeIndex or a 'timestamps' column.")

    # Define windows
    val_start = pd.Timestamp(val_start_str)
    val_end   = val_start + pd.Timedelta(days=val_days)
    test_start = val_end
    test_end   = test_start + pd.Timedelta(days=test_days)

    # Build masks (inclusive start, exclusive end)
    idx = np.arange(len(df))
    ts = df.index

    val_mask  = (ts >= val_start) & (ts < val_end)
    test_mask = (ts >= test_start) & (ts < test_end)

    # Ensure they are disjoint
    overlap = val_mask & test_mask
    if overlap.any():
        raise RuntimeError("Validation and test windows overlap; check the inputs.")

    train_mask = ~(val_mask | test_mask)

    train_idx = idx[train_mask]
    val_idx   = idx[val_mask]
    test_idx  = idx[test_mask]

    return train_idx, val_idx, test_idx

def smape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return np.mean(2.0 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100.0

class CNNLSTMWithFuture(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, cnn_out_channels=32, kernel_size=3):
        super().__init__()
        # self.norm = nn.LayerNorm(input_size) # Add normalization layer before Conv1D
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
        # x_seq = self.norm(x_seq)
        x_seq = x_seq.permute(0, 2, 1) 
        x_seq = self.relu(self.conv1d(x_seq))  
        x_seq = x_seq.permute(0, 2, 1) 
        
        # LSTM + output
        _, (hn, _) = self.lstm(x_seq)
        return self.fc(hn[-1]).squeeze(-1)
    
class RHForecastDataset(Dataset):
    def __init__(self, df, input_cols, target_col, window_size):
        self.df = df.reset_index(drop=True)
        self.input_cols = input_cols
        self.target_col = target_col
        self.window_size = window_size

    def __len__(self):
        return len(self.df) - self.window_size

    def __getitem__(self, idx):
        x = self.df.loc[idx:idx + self.window_size - 1, self.input_cols].values.astype(np.float32)
        y = self.df.loc[idx + self.window_size, self.target_col]
        return torch.tensor(x), torch.tensor(y, dtype=torch.float32)
        

COLORS = {
    "true":   "#0072B2",  # blue
    "pred":   "#E69F00",  # orange
    "resid":  "#CC79A7",  # magenta
    "eqline": "#4D4D4D",  # dark gray
    "zero":   "#8A8A8A",  # mid gray
}

def setup_matplotlib():
    plt.rcParams.update({
        "figure.dpi": 130,
        "figure.facecolor": "white",
        "axes.facecolor": "#FBFBFC",
        "axes.edgecolor": "#D0D0D0",
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "grid.color": "#E6E6E6",
        "grid.alpha": 0.6,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": True,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#D0D0D0",
        "font.size": 11,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "lines.linewidth": 2.2,     # a little bolder by default
        "axes.prop_cycle": cycler("color", [COLORS["true"], COLORS["pred"], COLORS["resid"]]),
})
    
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
        

class HVACOffDataset(Dataset):
    def __init__(self, df, input_cols, target_col='Tin_target', window_size=6):
        
        self.X_seq = []
        self.y = []
        self.is_test_flags = []

        data = df[input_cols + [target_col, 'is_test']].copy()

        arr = data[input_cols].values
        target = data[target_col].values
        flags = data['is_test'].values

        for i in range(window_size, len(data)):
            self.X_seq.append(arr[i - window_size:i])
            self.y.append(target[i])
            self.is_test_flags.append(flags[i])

        self.X_seq = torch.tensor(self.X_seq, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.float32)
        self.is_test_flags = np.array(self.is_test_flags)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx): return self.X_seq[idx], self.y[idx]


class CausalConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, k, dilation=1, dropout=0.2):
        super().__init__()
        self.pad = nn.ConstantPad1d(( (k-1)*dilation, 0 ), 0.0)  # left-only
        self.conv = nn.Conv1d(in_ch, out_ch, k, dilation=dilation, padding=0)
        self.drop = nn.Dropout(dropout)
        self.act = nn.ReLU()
    def forward(self, x):
        return self.drop(self.act(self.conv(self.pad(x))))

class TCNRegressor(nn.Module):
    def __init__(self, in_dim, out_dim=1, hidden=256, kernel_size=5, dropout=0.2, levels=5):
        super().__init__()
        layers, ch = [], in_dim
        for l in range(levels):
            dil = 2**l
            layers += [CausalConv1d(ch, hidden, kernel_size, dilation=dil, dropout=dropout)]
            ch = hidden
        self.net = nn.Sequential(*layers)
        self.fc = nn.Linear(hidden, out_dim)
    def forward(self, x):               # (B,T,D)
        x = x.transpose(1, 2)           # (B,D,T)
        y = self.net(x)[:, :, -1]       # last causal step
        return self.fc(y).squeeze(-1)
    
class CausalBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, dilation=1, dropout=0.15):
        super().__init__()
        pad = (k - 1) * dilation
        self.pad = nn.ConstantPad1d((pad, 0), 0.0)

        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=k, dilation=dilation, padding=0)
        self.norm = nn.LayerNorm(out_ch)
        self.act  = nn.ReLU()
        self.drop = nn.Dropout(dropout)

        # 1x1 conv for residual if channels mismatch
        self.res_proj = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def forward(self, x):  # x: (B, C, T)
        res = x if self.res_proj is None else self.res_proj(x)
        y = self.conv(self.pad(x))          # (B, C_out, T)
        # LayerNorm over channel dim → permute to (B,T,C)
        y = y.transpose(1, 2)
        y = self.norm(y)
        y = y.transpose(1, 2)
        y = self.act(y)
        y = self.drop(y)
        return y + res

class ResTCNRegressor(nn.Module):
    def __init__(self, in_dim, hidden=256, levels=6, kernel_size=7, dropout=0.15, out_dim=1):
        super().__init__()
        ch = in_dim
        blocks = []
        for l in range(levels):
            dil = 2 ** l  # 1,2,4,8,16,32
            blocks.append(CausalBlock(ch, hidden, k=kernel_size, dilation=dil, dropout=dropout))
            ch = hidden
        self.net = nn.Sequential(*blocks)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):         # x: (B, T, D)
        x = x.transpose(1, 2)     # (B, D, T)
        y = self.net(x)[:, :, -1] # last causal step
        return self.head(y).squeeze(-1)
    

import os

def _transform_with_freeze(X_np, scaler):
    """
    Scale only non-frozen columns; keep sin/cos columns as-is.
    Works for 2D shape (N, D). Assumes scaler.frozen_idx is set.
    """
    if not hasattr(scaler, "frozen_idx") or not scaler.frozen_idx:
        return scaler.transform(X_np)
    D = X_np.shape[1]
    frozen = set(scaler.frozen_idx)
    non_frozen = [j for j in range(D) if j not in frozen]
    X_out = X_np.copy()
    if non_frozen:
        X_out[:, non_frozen] = scaler.transform(X_np[:, non_frozen])
    return X_out

def _rf_vector_for_model(model, rf_map, default_names=None):
    """
    Build an input vector for a trained sklearn model using its own feature names.
    - Uses model.feature_names_in_ when available (preferred).
    - Falls back to provided default_names (e.g., saved with the model).
    - As a last resort, uses sorted(rf_map.keys()).
    """
    if model is None:
        return None, []
    names = list(getattr(model, "feature_names_in_", []))
    if not names:
        names = list(default_names) if default_names else sorted(rf_map.keys())
    x = [float(rf_map.get(c, 0.0)) for c in names]
    return x, names
import joblib

DEFAULT_AH_COLS = lambda hvac_col: [
    "AH","AH_lag1","AH_lag2","AH_lag3",
    "Tin","Tout","AH_out",
    hvac_col, "on_runtime_1h","is_trans",
    "hour_sin","hour_cos","month_sin","month_cos","doy_sin","doy_cos",
]


def _maybe_load_feature_list(json_path, fallback_cols):
    if os.path.exists(json_path):
        try:
            with open(json_path, "r") as f:
                data = json.load(f)
            cols = data.get("input_cols") or data.get("ah_input_cols") or data.get("cols")
            if cols: return list(cols)
        except Exception:
            pass
    return list(fallback_cols)

def _ensure_frozen_idx(scaler, cols):
    # freeze cyclical sines/cosines
    cyc = tuple(j for j, c in enumerate(cols) if c.endswith("_sin") or c.endswith("_cos"))
    if not hasattr(scaler, "frozen_idx") or scaler.frozen_idx is None:
        scaler.frozen_idx = cyc
    else:
        scaler.frozen_idx = tuple(sorted(set(tuple(scaler.frozen_idx)) | set(cyc)))

# def load_models_and_scalers(
#     df_index: int,
#     hvac_col: str = "hvac_mode",
#     ah_input_cols=None,
#     tin_off_input_cols=None,
#     model_dir_tpl: str = "models/df_{idx}",
#     scaler_dir_tpl: str = "models/scalers/df_{idx}",
#     device: str = "cuda" if torch.cuda.is_available() else "cpu",
#     CNNLSTMWithFuture=None,      # class
#     ResTCNRegressor=None,        # class
#     res_tcn_kwargs: dict | None = None
# ):
#     """
#     Loads:
#       - AH→RH model:  models/df_{idx}/best_ah.pt + models/scalers/df_{idx}/scaler_ah.pkl (+ ah_features.json)
#       - Tin-OFF TCN:  models/df_{idx}/best_tinv2.pt + models/scalers/df_{idx}/scaler_tin.pkl
#       - Tin-ON RF:    models/df_{idx}/rf_on_1.pkl / rf_on_2.pkl (or best_model_rf.pkl) + rf_on_features.json (optional)
#       - Optional solar correction: models/df_{idx}/calibration.json
#     """
#     assert CNNLSTMWithFuture is not None and ResTCNRegressor is not None, \
#         "Pass CNNLSTMWithFuture and ResTCNRegressor classes."

#     mdir = model_dir_tpl.format(idx=df_index)
#     sdir = scaler_dir_tpl.format(idx=df_index)
#     res_tcn_kwargs = res_tcn_kwargs or {}

#     # --- scalers ---
#     tin_scaler = joblib.load(os.path.join(sdir, "scaler_tin.pkl"))
#     ah_scaler  = joblib.load(os.path.join(sdir, "scaler_ah.pkl"))

#     # --- RF (Tin-ON) ---
#     rf_on_1 = rf_on_2 = rf_single = None
#     rf_features = None
#     fjson = os.path.join(mdir, "rf_on_features.json")
#     if os.path.exists(fjson):
#         try:
#             with open(fjson, "r") as f:
#                 rf_features = list(json.load(f).get("features", []))
#         except Exception:
#             rf_features = None

#     p1 = os.path.join(mdir, "rf_on_1.pkl")
#     p2 = os.path.join(mdir, "rf_on_2.pkl")
#     if os.path.exists(p1): rf_on_1 = joblib.load(p1)
#     if os.path.exists(p2): rf_on_2 = joblib.load(p2)

#     # single-RF fallback
#     p_single = os.path.join(mdir, "best_model_rf.pkl")
#     if os.path.exists(p_single):
#         rf_single = joblib.load(p_single)

#     # --- AH model (feature order from json or fallback) ---
#     def _maybe_load_feature_list(json_path, fallback_cols):
#         if os.path.exists(json_path):
#             try:
#                 with open(json_path, "r") as f:
#                     data = json.load(f)
#                 cols = data.get("input_cols") or data.get("ah_input_cols") or data.get("cols")
#                 if cols:
#                     return list(cols)
#             except Exception:
#                 pass
#         return list(fallback_cols)

#     ah_feat_json = os.path.join(mdir, "ah_features.json")
#     DEFAULT_AH = DEFAULT_AH_COLS(hvac_col)
#     ah_cols_saved = _maybe_load_feature_list(ah_feat_json, DEFAULT_AH)

#     ah_model = CNNLSTMWithFuture(
#         input_size=len(ah_cols_saved),
#         hidden_size=128, num_layers=2,
#         cnn_out_channels=32, kernel_size=3
#     ).to(device)
#     ah_state = torch.load(os.path.join(mdir, "best_ah.pt"), map_location=device)
#     ah_model.load_state_dict(ah_state, strict=True)
#     ah_model.eval()

#     # --- TCN OFF ---
#     assert tin_off_input_cols is not None, "Provide tin_off_input_cols to size the TCN."
#     tcn_tin_off = ResTCNRegressor(in_dim=len(tin_off_input_cols), **res_tcn_kwargs).to(device)
#     tcn_state = torch.load(os.path.join(mdir, "best_tinv2.pt"), map_location=device)
#     tcn_tin_off.load_state_dict(tcn_state, strict=True)
#     tcn_tin_off.eval()

#     # freeze cyclical cols in scalers (if not already set)
#     _ensure_frozen_idx(tin_scaler, tin_off_input_cols)
#     _ensure_frozen_idx(ah_scaler,  ah_cols_saved)

#     # optional solar correction
#     alpha_sw_off, alpha_hours, alpha_min_sw = 0.0, (10, 16), 50.0
#     calib_path = os.path.join(mdir, "calibration.json")
#     if os.path.exists(calib_path):
#         try:
#             with open(calib_path, "r") as f:
#                 calib = json.load(f)
#             alpha_sw_off = float(calib.get("alpha_sw_off", 0.0))
#             hrs = calib.get("hours", [10, 16])
#             alpha_hours = (int(hrs[0]), int(hrs[1])) if isinstance(hrs, (list, tuple)) else (10, 16)
#             alpha_min_sw = float(calib.get("min_sw", 50.0))
#         except Exception:
#             pass

#     return dict(
#         ah_model=ah_model,
#         ah_scaler=ah_scaler,
#         ah_input_cols=ah_cols_saved,
#         tcn_tin_off=tcn_tin_off,
#         tin_scaler=tin_scaler,
#         rf_on_1=rf_on_1,
#         rf_on_2=rf_on_2,
#         rf_single=rf_single,
#         rf_features=rf_features,
#         alpha_sw_off=alpha_sw_off,
#         alpha_hours=alpha_hours,
#         alpha_min_sw=alpha_min_sw,
#     )
_binaries = {"is_trans_off", "off_runtime_1h", "on_runtime_1h", "is_trans"}

def _compute_frozen_idx(cols):
    return tuple(i for i, c in enumerate(cols) if c.endswith(("_sin","_cos")) or c in _binaries)

def load_models_and_scalers(
    df_index: int,
    hvac_col: str = "hvac_mode",
    ah_input_cols=None,   
    tin_off_input_cols=None,
    model_dir_tpl: str = "models/df_{idx}",
    scaler_dir_tpl: str = "models/scalers/df_{idx}",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    CNNLSTMWithFuture=None,   # pass class
    ResTCNRegressor=None,       # ⬅️ use the new class
    res_tcn_kwargs: dict | None = None        # pass class
):
    """
    Loads:
      - RH model from:    models/df_{idx}/best_rh.pt
      - TCN (Tin-OFF):    models/df_{idx}/best_tinv2.pt
      - RF  (Tin-ON):     models/df_{idx}/best_model_rf.pkl
      - RH scaler:        models/scalers/df_{idx}/scaler_rh.pkl
      - TCN scaler (Tin): models/scalers/df_{idx}/scaler_tin.pkl
    Instantiates torch models with correct input sizes.
    """
    mdir = model_dir_tpl.format(idx=df_index)
    sdir = scaler_dir_tpl.format(idx=df_index)
    ah_obj = joblib.load(os.path.join(sdir, "scaler_ah.pkl"))
    if isinstance(ah_obj, dict) and "scaler" in ah_obj:
        ah_scaler = ah_obj["scaler"]
        ah_trained = list(ah_obj.get("feature_order", []) or [])
    else:
        ah_scaler = ah_obj
        ah_trained = []

    ah_feat_json = os.path.join(mdir, "ah_features.json")
    if not ah_trained and os.path.exists(ah_feat_json):
        with open(ah_feat_json, "r") as f:
            _data = json.load(f)

        if isinstance(_data, list):
            ah_trained = list(_data)
        elif isinstance(_data, dict):
            for k in ("features", "input_cols", "feature_order", "cols"):
                if k in _data:
                    ah_trained = list(_data[k])
                    break
        else:
            raise ValueError(
                f"{ah_feat_json} must be a list or a dict with one of "
                f"['features','input_cols','feature_order','cols']"
            )

    if not ah_trained or not all(isinstance(c, str) for c in ah_trained):
        raise ValueError(f"{ah_feat_json} does not contain a valid list of feature names.")
    # sanity
    if not ah_trained or not all(isinstance(c, str) for c in ah_trained):
        raise ValueError(f"{ah_feat_json} does not contain a valid list of feature names.")
    # OFF (Tin)
    _tin_obj = joblib.load(os.path.join(sdir, "scaler_tin.pkl"))  # <-- name aligned to OFF
    if not isinstance(_tin_obj, dict) or "feature_order" not in _tin_obj:
        raise ValueError("scaler_tin_off.pkl must be a dict with {'scaler', 'feature_order'}. Re-save from training.")
    tin_scaler = _tin_obj["scaler"]
    tin_off_trained = list(_tin_obj["feature_order"])


    ah_input_cols = ah_trained
    tin_off_input_cols = tin_off_trained
    
    if not hasattr(ah_scaler, "frozen_idx") or ah_scaler.frozen_idx is None:
        ah_scaler.frozen_idx = _compute_frozen_idx(ah_input_cols)

    if not hasattr(tin_scaler, "frozen_idx") or tin_scaler.frozen_idx is None:
        tin_scaler.frozen_idx = _compute_frozen_idx(tin_off_input_cols)

    rf_on_1 = rf_on_2 = rf_single = None
    rf_features = None

    fjson = os.path.join(mdir, "rf_on_features.json")
    if not os.path.exists(fjson):
        raise FileNotFoundError(f"Missing {fjson}. Save the exact feature order during RF training.")
    with open(fjson, "r") as f:
        data = json.load(f)
    rf_features = list(data) if isinstance(data, list) else list(data.get("features", []))
    if not rf_features:
        raise ValueError("rf_on_features.json is empty or malformed.")
    
    p1 = os.path.join(mdir, "rf_on_1.pkl")
    p2 = os.path.join(mdir, "rf_on_2.pkl")
    if os.path.exists(p1): rf_on_1 = joblib.load(p1)
    if os.path.exists(p2): rf_on_2 = joblib.load(p2)

    # absolute fallback to old single RF if present
    p_single = os.path.join(mdir, "best_model_rf.pkl")
    if os.path.exists(p_single):
        rf_single = joblib.load(p_single)

    # --- torch models
    assert CNNLSTMWithFuture is not None and ResTCNRegressor  is not None, \
        "Pass CNNLSTMWithFuture and ResTCNRegressor classes to load torch models."

    # RH
    assert ah_input_cols is not None and tin_off_input_cols is not None
    ah_model = CNNLSTMWithFuture(
        input_size=len(ah_input_cols),
        hidden_size=128, num_layers=2,
        cnn_out_channels=32, kernel_size=3
    ).to(device)
    ah_state = torch.load(os.path.join(mdir, "best_ah.pt"), map_location=device)
    
    expected_in, expected_out, expected_ks = None, None, None
    for k, v in ah_state.items():
        # look for a 1D conv weight tensor: (out_channels, in_channels, kernel_size)
        if k.endswith("weight") and isinstance(v, torch.Tensor) and v.ndim == 3:
            expected_out, expected_in, expected_ks = v.shape
            break
    if expected_in is None:
        raise RuntimeError("Could not infer conv1d in_channels from AH checkpoint; check layer names.")

    # strict: your feature count must match training
    n_ah = len(ah_input_cols)
    assert n_ah == expected_in, (
        f"AH feature count mismatch: trained expects {expected_in} features, but you provided {n_ah}. "
        f"Fix ah_input_cols to match training or re-train."
    )

    
    
    ah_model.load_state_dict(ah_state, strict=True)
    ah_model.eval()

    # TCN (Tin-OFF)
    assert tin_off_input_cols is not None, "Provide tin_off_input_cols to size TCN."
    tcn_tin_off =  ResTCNRegressor(in_dim=len(tin_off_input_cols), **res_tcn_kwargs).to(device)
    tcn_state = torch.load(os.path.join(mdir, "best_tinv2.pt"), map_location=device)
    tcn_tin_off.load_state_dict(tcn_state, strict=True)
    tcn_tin_off.eval()

    # keep frozen indices on scaler (used by your trainer)
    # if not hasattr(tin_scaler, "frozen_idx") or tin_scaler.frozen_idx is None:
    #     tin_scaler.frozen_idx = tuple(_indices_to_freeze(tin_off_input_cols))

    calib_path = os.path.join(model_dir_tpl.format(idx=df_index), "calibration.json")
    alpha_sw_off = 0.0
    alpha_hours = (10, 16)   # inclusive hours where α is allowed
    alpha_min_sw = 50.0      # W/m² threshold
    calib_path = os.path.join(model_dir_tpl.format(idx=df_index), "calibration.json")
    alpha_sw_off, alpha_hours, alpha_min_sw = 0.0, (10, 16), 50.0
    if os.path.exists(calib_path):
        try:
            with open(calib_path, "r") as f:
                calib = json.load(f)
            alpha_sw_off = float(calib.get("alpha_sw_off", 0.0))
            # tuples for safety
            hrs = calib.get("hours", [10, 16])
            alpha_hours = (int(hrs[0]), int(hrs[1])) if isinstance(hrs, (list, tuple)) else (10, 16)
            alpha_min_sw = float(calib.get("min_sw", 50.0))
        except Exception:
            alpha_sw_off, alpha_hours, alpha_min_sw = 0.0, (10, 16), 50.0

    # loaded["alpha_sw_off"] = alpha_sw_off
    return dict(
        ah_model=ah_model,
        ah_scaler=ah_scaler,
        tcn_tin_off=tcn_tin_off,
        tin_scaler=tin_scaler,
        rf_on_1=rf_on_1, 
        rf_on_2=rf_on_2, 
        rf_single=rf_single,
        rf_features=rf_features,   # ← add this
        alpha_sw_off=alpha_sw_off,
        alpha_hours=alpha_hours,
        alpha_min_sw=alpha_min_sw,
        ah_input_cols=ah_input_cols,
        tin_off_input_cols=tin_off_input_cols,
    )

def _rf_vector_for_model(model, rf_map, default_names=None):
    """
    Build an input vector for a trained sklearn model using its own feature names.
    - Uses model.feature_names_in_ when available (preferred).
    - Falls back to provided default_names (e.g., saved with the model).
    - As a last resort, uses sorted(rf_map.keys()).
    """
    if model is None:
        return None, []
    names = list(getattr(model, "feature_names_in_", []))
    if not names:
        names = list(default_names) if default_names else sorted(rf_map.keys())
    x = [float(rf_map.get(c, 0.0)) for c in names]
    return x, names


def _steps_per_hour(idx: pd.DatetimeIndex) -> int:
    """Infer samples per hour from the index (robust to missing .freq)."""
    if idx.freq is not None:
        off = pd.tseries.frequencies.to_offset(idx.freq)
        dt = off.delta if hasattr(off, "delta") else pd.to_timedelta(off.n, unit=off.name)
    else:
        dt = pd.to_timedelta(np.median(np.diff(idx.view("i8"))), unit="ns")
    return max(1, int(round(pd.Timedelta(hours=1) / dt)))

def prepare_tin_off_features(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Return df with extra leak-safe features and the input col list for OFF model."""
    g = df.copy()

    # 1-step trends (no lookahead)
    g["Tin_diff1"]  = g["Tin"].diff().fillna(0.0)
    g["Tout_diff1"] = g["Tout"].diff().fillna(0.0)

    # tiny smoothers (no lookahead)
    g["Tin_ma3"]  = g["Tin"].rolling(3, min_periods=1).mean()
    g["Tout_ma3"] = g["Tout"].rolling(3, min_periods=1).mean()

    # ensure Tout_fut1 exists; last value bfilled just for shape (never used in loss)
    if "Tout_fut1" not in g.columns:
        g["Tout_fut1"] = g["Tout"].shift(-1).bfill()

    # optional shortwave radiation if present
    # sw_cols = [c for c in ["SW_down", "SW1h", "SW3h"] if c in g.columns]
    if "hvac_mode" in g.columns:
        off_flag = (g["hvac_mode"] == 0).astype(int)
        g["is_trans_off"] = ((g["hvac_mode"] == 0) & (g["hvac_mode"].shift(1) != 0)).astype(int)
        g["off_runtime_1h"] = off_flag.rolling(_steps_per_hour(g.index), min_periods=1).sum()
    else:
        g["is_trans_off"] = 0
        g["off_runtime_1h"] = 0.0

    tin_off_input_cols = [
        "Tin","Tin_diff1","Tin_ma3",
        "Tout","Tout_diff1","Tout_ma3",
        "hour_sin","hour_cos","dow_sin","dow_cos",
        "Tout_fut1","SW_down","SW1h","SW3h","wind10m", "cloudcover", "is_trans_off","off_runtime_1h",
    ]
    # drop rows that lack any required inputs
    g = g.dropna(subset=[c for c in tin_off_input_cols if c in g.columns])

    return g, tin_off_input_cols

def forecast_next_96(
    df: pd.DataFrame,
    window_start_idx: int,
    df_index: int = 0,
    hvac_col: str = "hvac_mode",
    ah_input_cols = None,         
    tin_off_input_cols = None,  
    win_ah: int = 24,
    win_tin: int = 48,
    horizon: int = 96,
    # Either pass models/scalers or let me load them from disk:
    ah_model=None, ah_scaler=None,
    tcn_tin_off=None, tin_scaler=None,
    rf_on_1=None, rf_on_2=None, rf_single=None,
    load_from_disk: bool = True,
    loaded_dict = None,
    model_dir_tpl: str = "models/df_{idx}",
    scaler_dir_tpl: str = "models/scalers/df_{idx}",
    # For torch
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    # Classes for optional loading
    CNNLSTMWithFuture=None,
    ResTCNRegressor=None,            # ⬅️ new param
    res_tcn_kwargs: dict | None = None,
):
    """
    Outputs: DataFrame indexed by timestamp with columns:
      Tin_pred, Tin_true, RH_pred, RH_true, hvac_mode, Tin_target
    """   

    assert ah_input_cols is not None and tin_off_input_cols is not None, "Provide ah_input_cols & tin_off_input_cols."

    # Safety: enough history and horizon present
    need_hist = max(win_ah, win_tin, 3)  # need >=3 to form MA3/diff features
    assert window_start_idx >= need_hist, f"Need at least {need_hist} history rows before window_start_idx."
    assert (window_start_idx + horizon) <= len(df), "DataFrame must include the full future horizon."
    g = df.copy()

    def _steps_per_hour(idx: pd.DatetimeIndex) -> int:
        if idx.freq is not None:
            off = pd.tseries.frequencies.to_offset(idx.freq)
            step = off.delta if hasattr(off, "delta") else pd.to_timedelta(off.n, unit=off.name)
        else:
            step = idx.to_series().diff().median()
        if pd.isna(step) or step == pd.Timedelta(0):
            return 4  # assume 15-min
        return int(round(pd.Timedelta(hours=1) / step))

    steps_per_h = _steps_per_hour(g.index)
    # Load if requested
    if load_from_disk:
        loaded = load_models_and_scalers(
            df_index=df_index,
            hvac_col=hvac_col,
            ah_input_cols=ah_input_cols,
            tin_off_input_cols=tin_off_input_cols,
            model_dir_tpl=model_dir_tpl,
            scaler_dir_tpl=scaler_dir_tpl,
            device=device,
            CNNLSTMWithFuture=CNNLSTMWithFuture,
            ResTCNRegressor=ResTCNRegressor,      
            res_tcn_kwargs=res_tcn_kwargs,
        )
        ah_model   = loaded["ah_model"]
        ah_scaler  = loaded["ah_scaler"]
        tcn_tin_off= loaded["tcn_tin_off"]
        tin_scaler = loaded["tin_scaler"]
        # rf_tin_on  = loaded["rf_tin_on"]
        rf_features = loaded['rf_features']
        alpha_sw_off = loaded['alpha_sw_off']
        alpha_hours   = loaded['alpha_hours']
        alpha_min_sw  = loaded['alpha_min_sw']
        rf_on_1     = loaded["rf_on_1"]
        rf_on_2     = loaded["rf_on_2"]
        rf_single   = loaded["rf_single"]
        ah_input_cols = loaded["ah_input_cols"]
        tin_off_input_cols = loaded["tin_off_input_cols"]
    else:
        ah_model   = loaded_dict["ah_model"]
        ah_scaler  = loaded_dict["ah_scaler"]
        tcn_tin_off= loaded_dict["tcn_tin_off"]
        tin_scaler = loaded_dict["tin_scaler"]
        # rf_tin_on  = loaded_dict["rf_tin_on"]
        rf_features = loaded_dict['rf_features']
        alpha_sw_off = loaded_dict['alpha_sw_off']
        alpha_hours   = loaded_dict['alpha_hours']
        alpha_min_sw  = loaded_dict['alpha_min_sw']
        rf_on_1     = loaded_dict["rf_on_1"]
        rf_on_2     = loaded_dict["rf_on_2"]
        rf_single   = loaded_dict["rf_single"]
        ah_input_cols = loaded_dict["ah_input_cols"]
        tin_off_input_cols = loaded_dict["tin_off_input_cols"]

    # Put torch models on device/eval
    if hasattr(ah_model, "to"): ah_model.to(device).eval()
    if hasattr(tcn_tin_off, "to"): tcn_tin_off.to(device).eval()

    def _need(df, cols, name):
        miss = [c for c in cols if c not in df.columns]
        assert not miss, f"[{name}] missing columns: {miss}"
    
    _need(g, ["AH","AH_lag1","AH_lag2","AH_lag3"], "AH lags")

    # ---------- Seed history windows (STRICTLY past rows) ----------
    # Tin-OFF sequence (scaled with freeze)
    tin_seq = []
    for k in range(window_start_idx - win_tin, window_start_idx):
        r = g.iloc[k]
        missing = [c for c in tin_off_input_cols if c not in r.index]
        assert not missing, f"[Tin-OFF seed] row {k} missing: {missing}"
        vals = [float(r[c]) for c in tin_off_input_cols]
        v = np.asarray(vals, dtype=np.float32)[None, :]
        v_s = _transform_with_freeze(v, tin_scaler)
        tin_seq.append(v_s[0].tolist())

    def _make_ah_row_at(k, tin_val=None, ah_prev=None, ah_l1=None, ah_l2=None, ah_l3=None):
        """Create an AH feature row for position k (using df values for seed)."""
        r = g.iloc[k]
        mp = {
            "AH":      float(ah_prev if ah_prev is not None else r["AH"]),
            "AH_lag1": float(ah_l1   if ah_l1   is not None else r["AH_lag1"]),
            "AH_lag2": float(ah_l2   if ah_l2   is not None else r["AH_lag2"]),
            "AH_lag3": float(ah_l3   if ah_l3   is not None else r["AH_lag3"]),
            "Tin":     float(tin_val),            
            "Tout":    float(r["Tout"]),
            "AH_out":  float(r["AH_out"]),
            hvac_col:  float(r[hvac_col]),
            "on_runtime_1h": float(r["on_runtime_1h"]),
            "is_trans":      float(r["is_trans"]),
            "hour_sin":  float(r["hour_sin"]), 
            "hour_cos":  float(r["hour_cos"]),
            "month_sin": float(r["month_sin"]), 
            "month_cos": float(r["month_cos"]),
        }
        return [float(mp.get(c, r.get(c, 0.0))) for c in ah_input_cols]
    # RH sequence (mixed scaling: only Tout & hvac_col)
    # scaling_cols_rh = [c for c in ah_input_cols if not (c.startswith('hour_') or c.startswith('RH') or c.startswith('month'))]

    # Buffers for autoregressive features
    # Tin buffer holds [Tin_{t-2}, Tin_{t-1}, Tin_t] at t = window_start_idx
    tin_buf  = [
        float(g["Tin"].iloc[window_start_idx-2]),
        float(g["Tin"].iloc[window_start_idx-1]),
        float(g["Tin"].iloc[window_start_idx]),
    ]
    # Tout buffer holds [Tout_{t-2}, Tout_{t-1}] so we can form diff/ma3 with current Tout
    tout_buf = [
        float(g["Tout"].iloc[window_start_idx-2]),
        float(g["Tout"].iloc[window_start_idx-1]),
    ]
    tin_prev = float(g["Tin"].iloc[window_start_idx]) 

    # Seed RH lags at t0
    AH_prev = float(g["AH"].iloc[window_start_idx])
    AH_l1   = float(g["AH"].shift(1).iloc[window_start_idx])
    AH_l2   = float(g["AH"].shift(2).iloc[window_start_idx])
    AH_l3   = float(g["AH"].shift(3).iloc[window_start_idx])

    ah_seq = []
    for k in range(window_start_idx - win_ah, window_start_idx):
        row = _make_ah_row_at(k, tin_val=float(g["Tin"].iloc[k]))   # <-- no comma
        v = np.asarray(row, dtype=np.float32).reshape(1, -1)        # always 2-D
        v_s = _transform_with_freeze(v, ah_scaler)
        ah_seq.append(v_s[0].tolist())

    results = []

    _need(g, tin_off_input_cols, "Tin-OFF")
    _need(g, rf_features,        "Tin-ON")
    _need(g, ah_input_cols,      "AH->RH")
    # exogenous required for this notebook
    _need(g, ["Tout_fut1","AH_out"], "exogenous")
    # ---------- Recursive rollout ----------
    with torch.no_grad():
        for step in range(horizon):
            i = window_start_idx + step
            r = g.iloc[i]

            hv_mode = int(r[hvac_col])
            Tout    = float(r["Tout"])
            hour_sin = float(r.get("hour_sin", 0.0))
            hour_cos = float(r.get("hour_cos", 0.0))
            dow_sin  = float(r.get("dow_sin", 0.0))
            dow_cos  = float(r.get("dow_cos", 0.0))
            month_sin= float(r.get("month_sin", 0.0))
            month_cos= float(r.get("month_cos", 0.0))
            Tout_fut1 = float(g["Tout_fut1"].iloc[i])

            # ---- Tin step
            if hv_mode == 0:
                # Build complete OFF feature row using buffers (no leakage)
                Tin_now   = tin_buf[-1]          # Tin_t (previous known/pred)
                Tin_prev1 = tin_buf[-2]
                Tin_prev2 = tin_buf[-3]
                Tin_diff = Tin_now - Tin_prev1
                Tin_ma3   = float((Tin_now + Tin_prev1 + Tin_prev2)/3.0)

                Tout_prev1 = tout_buf[-1]
                Tout_prev2 = tout_buf[-2]
                Tout_diff = Tout - Tout_prev1
                Tout_ma3   = float((Tout + Tout_prev1 + Tout_prev2)/3.0)

                off_map = {
                    "Tin": Tin_now,
                    "Tin_diff": Tin_diff,
                    "Tin_ma3": Tin_ma3,
                    "Tout": Tout,
                    "Tout_diff": Tout_diff,
                    "Tout_ma3": Tout_ma3,
                    "hour_sin": hour_sin, 
                    "hour_cos": hour_cos,
                    "Tout_fut1": Tout_fut1,
                }

                for c in tin_off_input_cols:
                    if c not in off_map:
                        off_map[c] = float(r[c])
                next_row = [float(off_map[c]) for c in tin_off_input_cols]
                v = np.asarray(next_row, dtype=np.float32)[None, :]
                v_s = _transform_with_freeze(v, tin_scaler)
                tin_seq = tin_seq[1:] + [v_s[0].tolist()]
                x_tin = torch.tensor([tin_seq], dtype=torch.float32, device=device)  # (1,T,D)
                tin_pred = float(tcn_tin_off(x_tin).detach().cpu().item())
                if alpha_sw_off:
                    sw_now = float(r.get("SW_down"))
                    hr     = int(pd.Timestamp(g.index[i]).hour)
                    if (alpha_hours[0] <= hr <= alpha_hours[1]) and np.isfinite(sw_now) and (sw_now >= alpha_min_sw):
                        tin_pred += alpha_sw_off * r.get("SW_down")

            else:
                # RF (Tin-ON): raw, no scaling
                Tin_now    = tin_buf[-1]      # Tin_i
                Tin_prev1  = tin_buf[-2]      # Tin_{i-1}
                Tin_prev2  = tin_buf[-3]      # Tin_{i-2}
                Tout_now   = Tout             # Tout_i
                Tout_prev1 = tout_buf[-1]     # Tout_{i-1}
                Tout_prev2 = tout_buf[-2]     # Tout_{i-2}

                Tin_diff  = Tin_now  - Tin_prev1
                Tout_diff = Tout_now - Tout_prev1
                Tin_ma3    = (Tin_now + Tin_prev1 + Tin_prev2) / 3.0
                Tout_ma3   = (Tout_now + Tout_prev1 + Tout_prev2) / 3.0

                rf_map = {
                    "Tin": Tin_now,
                    hvac_col: float(hv_mode),
                    "Tout": Tout,
                    "T_diff": Tout - Tin_now,
                    "Tin_diff": float(Tin_diff),
                    "Tout_diff": float(Tout_diff),
                    "Tin_ma3": float(Tin_ma3),
                    "Tout_ma3": float(Tout_ma3),
                    "hour_sin": float(hour_sin),
                    "hour_cos": float(hour_cos),
                    "Tout_fut1": float(Tout_fut1),
                }
                for c in rf_features:
                    if c not in rf_map:
                        rf_map[c] = float(r[c])

                mdl = {1: rf_on_1, 2: rf_on_2}.get(hv_mode) or rf_on_1 or rf_on_2 or rf_single
                x_rf, _used = _rf_vector_for_model(mdl, rf_map, default_names=rf_features)
                tin_pred = float(mdl.predict([x_rf])[0]) 

            tin_prev = tin_pred  # keep for RF path

            # Update Tin/Tout buffers AFTER prediction (for next-step engineered features)
            tin_buf = [tin_buf[-2], tin_buf[-1], tin_pred]
            tout_buf = [tout_buf[-1], Tout]

            # ---- RH step (sequence; only Tout & hvac_col scaled, with RH lag buffer)
            # hv_val = float(hv_mode) if hvac_col == "hvac_mode" else float(r[hvac_col])

            ah_row = _make_ah_row_at(
                i,
                tin_val=tin_pred,           # <-- use predicted Tin at this step
                ah_prev=AH_prev, ah_l1=AH_l1, ah_l2=AH_l2, ah_l3=AH_l3
            )
            v = np.asarray(ah_row, dtype=np.float32)[None, :]
            v_s = _transform_with_freeze(v, ah_scaler)
            ah_seq = ah_seq[1:] + [v_s[0].tolist()]
            x_ah = torch.tensor([ah_seq], dtype=torch.float32, device=device)
            ah_pred = float(ah_model(x_ah).detach().cpu().item())

            # advance AH lag buffer
            AH_l3, AH_l2, AH_l1, AH_prev = AH_l2, AH_l1, AH_prev, ah_pred

            # convert AH_pred + Tin_pred -> RH_pred (%)
            RH_pred = RH_percent_from_AH_T(ah_pred, tin_pred)

            # ---- collect
            results.append({
                "timestamp": g.index[i],
                "Tin_pred": tin_pred,
                "Tin_true": float(r["Tin"]) if "Tin" in r else np.nan,
                "RH_pred": RH_pred,
                "RH_true": float(r["RH"]) if "RH" in r else np.nan,
                "hvac_mode": hv_mode,
                "Tin_target": float(r["Tin_target"]) if "Tin_target" in r else np.nan,
            })

    out = pd.DataFrame(results).set_index("timestamp")
    return out
