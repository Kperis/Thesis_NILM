import pandas as pd
import os, json, requests
import numpy as np

FEATURES_TIN_OFF = [
    # core
    "Tin", "Tin_diff", "Tin_ma3",
    # weather
    "Tout", "Tout_diff", "Tout_ma3",
    # time encodings
    "hour_sin", "hour_cos",
    # meteo/solar (adjust to what you actually use)
    "SW1h", "SW3h", 
    # future Tout (ensure this is a forecast at inference)
    "Tout_fut1",
    # flags/heuristics (if used)
    "is_trans_off", "off_runtime_1h",
]

FEATURES_TIN_ON = [
    "Tin", "Tout", "T_diff",
    "Tin_diff", "Tout_diff", "Tin_ma3", "Tout_ma3",
    "hour_sin", "hour_cos",
    "Tout_fut1",
    "SW_down", "SW1h", "SW3h",
]

OPENMETEO_ERA5 = "https://archive-api.open-meteo.com/v1/era5"

# ---------- freq helpers ----------

def _es_hPa_from_Tc(Tc: float) -> float:
    """Saturation vapor pressure in hPa at air temperature Tc (°C)."""
    return 6.112 * np.exp((17.67 * Tc) / (Tc + 243.5))

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

    # Preserve pandas index if input is a Series
    if isinstance(Tc, pd.Series):
        return pd.Series(AH, index=Tc.index, name="AH")
    return AH

def RH_percent_from_AH_T(AH_gm3: float, Tc: float) -> float:
    """Relative humidity (%) from absolute humidity (g/m³) and temperature (°C)."""
    es = _es_hPa_from_Tc(Tc)
    # vapor pressure from AH
    e = max(0.0, AH_gm3) * (Tc + 273.15) / 216.7
    RH = 100.0 * (e / max(1e-6, es))
    return float(np.clip(RH, 0.0, 100.0))

def AH_to_RH(Tc, AH):
    """RH[%] back from AH[g/m^3] and T[°C]."""
    Tc = np.asarray(Tc, dtype=float)
    AH = np.asarray(AH, dtype=float)
    es = _es_hPa_from_Tc(Tc)
    RH = 100.0 * (AH * (Tc + 273.15) / 216.7) / es
    return np.clip(RH, 0.0, 100.0)

def _infer_freq_and_steps(idx: pd.DatetimeIndex) -> tuple[pd.Timedelta, int, int]:
    """
    Infer native frequency (Timedelta) and step counts for ~1h and ~3h rolling windows.
    """
    if idx.freq is not None:
        off = pd.tseries.frequencies.to_offset(idx.freq)
        # pandas offsets may not expose .delta on all types
        try:
            freq = off.delta
        except Exception:
            freq = pd.to_timedelta(off.n, unit=off.name)
    else:
        # robust fallback: median spacing
        freq = idx.to_series().diff().median()
    if pd.isna(freq) or freq <= pd.Timedelta(0):
        raise ValueError("Cannot infer a valid frequency from the index.")
    step_1h = max(1, int(round(pd.Timedelta(hours=1) / freq)))
    step_3h = max(1, int(round(pd.Timedelta(hours=3) / freq)))
    return freq, step_1h, step_3h


# ---------- core fetch & align ----------

def _fetch_openmeteo_block(start_date: str, end_date: str, lat: float, lon: float, tz: str) -> pd.DataFrame:
    """
    Fetch hourly ERA5 variables we need, return a DataFrame indexed by (tz-local) timestamps.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "timezone": tz,
        "hourly": ",".join([
            "shortwave_radiation",   
            "wind_speed_10m",        
            "wind_gusts_10m",        
            "cloud_cover",           
            "relative_humidity_2m",  
            "dew_point_2m",          
            "precipitation",        
            "temperature_2m",        
        ]),
    }
    r = requests.get(OPENMETEO_ERA5, params=params, timeout=60)
    r.raise_for_status()
    js = r.json()
    if "hourly" not in js or "time" not in js["hourly"]:
        raise RuntimeError(f"Unexpected Open-Meteo response keys: {list(js.keys())}")

    H = js["hourly"]
    t = pd.to_datetime(H["time"])
    dfh = pd.DataFrame(index=t)
    # map API names -> our column names
    rename = {
        "shortwave_radiation": "SW_hourly_Wm2",
        "wind_speed_10m": "wind10m",
        "wind_gusts_10m": "windgusts10m",
        "cloud_cover": "cloudcover",
        "relative_humidity_2m": "rh2m",
        "dew_point_2m": "dewpoint2m",
        "precipitation": "precip1h",
        "temperature_2m": "Tout_hourly",
    }
    for k_api, k_out in rename.items():
        if k_api in H:
            dfh[k_out] = H[k_api]
    return dfh

def fetch_openmeteo_meteo(
    df: pd.DataFrame,
    lat: float = 43.0377,
    lon: float = -76.1330,
    tz: str = "America/New_York",
    cache_dir: str = "cache/openmeteo",
) -> pd.DataFrame:
    
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex")

    t0 = pd.to_datetime(df.index.min()).floor("D")
    t1 = pd.to_datetime(df.index.max()).ceil("D")
    start_date = (t0 - pd.Timedelta(days=1)).date().isoformat()
    end_date   = (t1 + pd.Timedelta(days=1)).date().isoformat()

    os.makedirs(cache_dir, exist_ok=True)
    cache_key = f"era5_all_{lat:.4f}_{lon:.4f}_{start_date}_{end_date}_{tz.replace('/','-')}.parquet"
    cache_path = os.path.join(cache_dir, cache_key)

    if os.path.exists(cache_path):
        hourly = pd.read_parquet(cache_path)
        hourly.index = pd.to_datetime(hourly.index)
    else:
        hourly = _fetch_openmeteo_block(start_date, end_date, lat, lon, tz)
        hourly.to_parquet(cache_path)

    # Align to df cadence
    freq, step_1h, step_3h = _infer_freq_and_steps(df.index)
    # full time range at target freq
    rng = pd.date_range(hourly.index.min(), hourly.index.max(), freq=freq)

    # interpolate all numeric columns to target cadence
    hourly_up = hourly.reindex(rng)
    # time-based interpolation for continuous meteo; then clamp known physical bounds
    hourly_up = hourly_up.interpolate(method="time")

    # clip negative radiation/precip
    if "SW_hourly_Wm2" in hourly_up:
        hourly_up["SW_hourly_Wm2"] = hourly_up["SW_hourly_Wm2"].clip(lower=0.0)
    if "precip1h" in hourly_up:
        hourly_up["precip1h"] = hourly_up["precip1h"].clip(lower=0.0)

    # reindex exactly to df (nearest within half a step), fill residual edges
    met = hourly_up.reindex(df.index, method="nearest", tolerance=freq/2).ffill().bfill()
    if "AH_out" not in met.columns:
        if "Tout_hourly" in met.columns and "rh2m" in met.columns:
            # uses your vectorized helper (already in this file)
            met["AH_out"] = AH_gm3_from_T_RH(met["Tout_hourly"], met["rh2m"])
        else:
            raise ValueError("Cannot compute AH_out: need Tout_hourly and rh2m.")
    # Shortwave roll-ups at native cadence
    cols = {}
    if "SW_hourly_Wm2" in met:
        SW_down = met["SW_hourly_Wm2"].rename("SW_down")
        SW1h = SW_down.rolling(step_1h, min_periods=max(1, step_1h//2)).mean().rename("SW1h")
        SW3h = SW_down.rolling(step_3h, min_periods=max(1, step_3h//2)).mean().rename("SW3h")
        cols.update({"SW_down": SW_down, "SW1h": SW1h, "SW3h": SW3h})

    for k in ["wind10m", "windgusts10m", "cloudcover", "rh2m", "dewpoint2m", "AH_out", "precip1h", "Tout_hourly"]:
        if k in met:
            cols[k] = met[k]

    out = pd.concat(cols.values(), axis=1) if cols else pd.DataFrame(index=df.index)
    return out

# ---------- public: enrich dataframe ----------

def add_meteo_features(
    df: pd.DataFrame,
    lat: float = 43.0377, lon: float = -76.1330, tz: str = "America/New_York",
    cache_dir: str = "cache/openmeteo",
) -> pd.DataFrame:
    """
    Returns a COPY of df with meteo & time features merged:
      - SW_down, SW1h, SW3h
      - wind10m, windgusts10m, cloudcover
      - rh2m, dewpoint2m, AH_out
      - precip1h
      - (optional) hour_sin/hour_cos, month_sin/month_cos
      - (optional) doy_sin/doy_cos
      - (optional) Tout_fut1 (shift -1)
    """
    g = df.copy()

    # add meteo block
    met = fetch_openmeteo_meteo(g, lat=lat, lon=lon, tz=tz, cache_dir=cache_dir)
    for c in met.columns:
        g[c] = met[c]
    return g


# Binary columns to be frozen (not scaled). You can list explicitly or infer below.
BINARY_COL_HINTS = {"is_trans_off", "off_runtime_1h", "is_trans", "on_runtime_1h"} 

def _steps_per_hour(index: pd.DatetimeIndex) -> int:
    if index.freq is not None:
        dt = index.freq.delta
    else:
        dt = pd.to_timedelta(np.median(np.diff(index.view("i8"))), unit="ns")
    return max(1, int(round(pd.Timedelta(hours=1) / dt)))

def get_tin_off_features(df):
    g = df.copy()
    return [c for c in FEATURES_TIN_OFF if c in g.columns]


def feature_engineering(df, hvac_col):
    g = df.copy()

    # Step 1: Try to set datetime index using 'timestamps' column
    if not isinstance(g.index, pd.DatetimeIndex):
        if 'timestamps' in g.columns:
            g['timestamps'] = pd.to_datetime(g['timestamps'], errors='coerce')
            g = g.set_index('timestamps')
        else:
            raise KeyError(f"DataFrame is missing 'timestamps' column. Available columns: {g.columns.tolist()}")
    # Step 2: Add cyclic time features
    g['hour'] = g.index.hour
    g['dow'] = g.index.dayofweek
    g['month'] = g.index.month
    g['hour_sin'] = np.sin(2 * np.pi * g['hour'] / 24)
    g['hour_cos'] = np.cos(2 * np.pi * g['hour'] / 24)
    g['dow_sin']  = np.sin(2 * np.pi * g['dow'] / 7)
    g['dow_cos']  = np.cos(2 * np.pi * g['dow'] / 7)
    g['month_sin'] = np.sin(2 * np.pi * g['month'] / 12)
    g['month_cos'] = np.cos(2 * np.pi * g['month'] / 12)
    g = add_meteo_features(
        g, lat=43.0377, lon=-76.1330, tz="America/New_York"
    )
    g['Tin_target'] = g['Tin'].shift(-1).fillna(20.0)
    g['RH_lag1'] = g['RH'].shift(1)
    g['RH_lag2'] = g['RH'].shift(2)
    g['RH_lag3'] = g['RH'].shift(3)
    g['RH_lag4'] = g['RH'].shift(4)
    g['RH_target'] = g['RH'].shift(-1)
    g["T_diff"]      = (g["Tout"] - g["Tin"])
    g["Tin_diff"]   = g["Tin"].diff().fillna(0.0)
    g["Tout_diff"]  = g["Tout"].diff().fillna(0.0)
    g["Tout_lag1"] = g["Tout"].shift(1).fillna(method="bfill")
    g["Tout_fut1"] = g["Tout"].shift(-1).fillna(method="ffill")
    g["Tin_ma3"]     = g["Tin"].rolling(3, min_periods=1).mean()
    g["Tout_ma3"]    = g["Tout"].rolling(3, min_periods=1).mean()  
    g["AH"]       = AH_gm3_from_T_RH(g["Tin"], g["RH"])
    g["AH_lag1"]  = g["AH"].shift(1)
    g["AH_lag2"]  = g["AH"].shift(2)
    g["AH_lag3"]  = g["AH"].shift(3)
    g["AH_target"] = g["AH"].shift(-1)        
    sp1h = _steps_per_hour(g.index)
    off_flag = (g[hvac_col] == 0).astype(int) if hvac_col in g.columns else 0
    g["is_trans_off"] = (((g[hvac_col]==0) & (g[hvac_col].shift(1)!=0)).astype(int)
                        if hvac_col in g.columns else 0)
    g["off_runtime_1h"] = (off_flag.rolling(sp1h, min_periods=1).sum()
                        if isinstance(off_flag, pd.Series) else 0)
    g["is_trans"] = (g[hvac_col] != g[hvac_col].shift(1)).astype(int).fillna(0)

    on_flag = (g[hvac_col] != 0).astype(int)
    g["on_runtime_1h"] = on_flag.rolling(4, min_periods=1).mean()
    g.fillna(method='bfill', inplace=True)
    return g