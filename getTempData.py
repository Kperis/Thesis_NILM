import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta, timezone

def ensure_tout_openmeteo(
    df: pd.DataFrame,
    start_idx: int,
    col_time: str = "timestamps",
    col_tout: str = "Tout",
    lat: float = 43.0377,      # Syracuse University, NY
    lon: float = -76.1349,     # Syracuse University, NY
) -> pd.DataFrame:
   
    if col_time not in df.columns:
        raise KeyError(f"Column '{col_time}' not found in df")
    if col_tout not in df.columns:
        # Create the column if missing so we can fill it
        df = df.copy()
        df[col_tout] = np.nan

    # Work on a copy to avoid in-place surprises
    out = df.copy()

    # Parse the window start time
    try:
        t0 = pd.to_datetime(out.loc[start_idx, col_time])
    except Exception as e:
        raise ValueError(f"start_idx {start_idx} invalid or '{col_time}' not parseable: {e}")

    # Normalize to UTC-naive for stable comparisons (Open-Meteo will return UTC)
    if pd.api.types.is_datetime64_any_dtype(out[col_time]) is False:
        out[col_time] = pd.to_datetime(out[col_time])

    # Build the 96-step 15-min target range
    target_times = pd.date_range(start=t0, periods=96, freq="15min")

    # Ensure all 96 timestamps exist as rows
    exist_mask = out[col_time].isin(target_times)
    missing_times = target_times.difference(out.loc[exist_mask, col_time])

    if len(missing_times) > 0:
        # Create empty rows for missing timestamps (all NaN except timestamp)
        add_rows = pd.DataFrame({col_time: missing_times})
        for c in out.columns:
            if c != col_time:
                add_rows[c] = np.nan
        out = pd.concat([out, add_rows], ignore_index=True)

    # Re-sort and reindex by time to make selection easy
    out.sort_values(col_time, inplace=True)
    out.reset_index(drop=True, inplace=True)

    # Now slice the exact 96-row window
    window_idx = out[col_time].isin(target_times)
    window = out.loc[window_idx, [col_time, col_tout]].copy()

    # If the selection somehow didn’t return 96 rows (e.g., duplicates), reindex explicitly
    window = (
        window.set_index(col_time)
              .reindex(target_times)  # enforce exactly those 96 stamps
              .rename_axis(col_time)
              .reset_index()
    )

    # If Tout is fully present (no NaNs), we’re done—merge back and return
    if window[col_tout].notna().all():
        return out

    # ===== Fetch from Open-Meteo (hourly) and upsample to 15-min =====
    # We fetch from t0.floor('D') through (t0 + 1 day).ceil('D') to safely cover the whole 24h window
    day_start = pd.Timestamp(t0.floor("D"), tz=timezone.utc)
    day_end = pd.Timestamp((t0 + pd.Timedelta(days=1)).floor("D"), tz=timezone.utc)

    # Open-Meteo has two endpoints: archive (past) and forecast (future).
    # We'll choose based on whether the day_start date is in the past or not (UTC).
    today_utc = pd.Timestamp(datetime.now(timezone.utc).date(), tz=timezone.utc)

    start_date_str = day_start.strftime("%Y-%m-%d")
    end_date_str = day_end.strftime("%Y-%m-%d")

    if day_start < today_utc:
        # Historical
        base_url = "https://archive-api.open-meteo.com/v1/archive"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "hourly": "temperature_2m",
            "timezone": "UTC",
        }
    else:
        # Forecast
        base_url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat,
            "longitude": lon,
            "start_date": start_date_str,
            "end_date": end_date_str,
            "hourly": "temperature_2m",
            "timezone": "UTC",
        }

    try:
        r = requests.get(base_url, params=params, timeout=20)
        r.raise_for_status()
        js = r.json()
        hourly = js.get("hourly", {})
        times = hourly.get("time", [])
        temps = hourly.get("temperature_2m", [])
        if not times or not temps or len(times) != len(temps):
            raise RuntimeError("Open-Meteo returned empty or mismatched hourly arrays.")
        hourly_df = pd.DataFrame({"time": pd.to_datetime(times), "temperature_2m": temps})
        hourly_df = hourly_df.set_index("time").sort_index()

        # Upsample to 15-min by time interpolation
        fifteen_df = (
            hourly_df
            .reindex(pd.date_range(hourly_df.index.min(), hourly_df.index.max(), freq="15min"))
            .interpolate(method="time")
        )

        # Restrict exactly to our 96 target times (in UTC)
        # Ensure target_times are treated as UTC-naive for matching; convert both to UTC-naive
        # since we constructed target_times tz-naive above
        src = fifteen_df.copy()
        src.index = pd.DatetimeIndex(src.index.values).tz_localize(None)

        fill_series = src.loc[src.index.isin(target_times), "temperature_2m"]

        # If any of the target times weren’t covered (e.g., t0 not aligned), reindex and interpolate again
        if len(fill_series.index) < 96:
            # Make sure we have values at ALL target times
            tmp = src.reindex(target_times).interpolate(method="time", limit_direction="both")
            fill_series = tmp["temperature_2m"]
    except Exception as e:
        raise RuntimeError(f"Failed to fetch/prepare Open-Meteo data: {e}")

    # Assign into window where Tout is NaN (or overwrite entirely if you prefer)
    need_mask = window[col_tout].isna()
    window.loc[need_mask, col_tout] = fill_series.values

    # Merge the filled window back into the output df
    out = out.set_index(col_time)
    window = window.set_index(col_time)
    out.loc[window.index, col_tout] = window[col_tout]
    out = out.reset_index()

    return out