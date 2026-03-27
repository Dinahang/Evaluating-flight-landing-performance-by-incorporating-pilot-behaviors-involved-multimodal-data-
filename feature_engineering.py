import glob
import math
import os
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
import pandas as pd

from pipeline_config import (  # type: ignore
    DATA_FOLDER,
    OUTPUT_FOLDER,
    SCENARIO_KEYWORDS,
    RUNWAY_TRUE_HEADING_DEG,
    RUNWAY_ELEVATION_FT,
    QNH_HPA,
    HPA_TO_FEET,
    N_SECONDS_BEFORE_TD,
    FALLBACK_DT_SECONDS,
    HARD_LANDING_FPM_THRESHOLD,
    CTRL_MAX_ABS,
    THROTTLE_CANDIDATES,
    AOA_CANDIDATES,
    BARO_ALT_CANDIDATES,
    BANK_CANDIDATES,
    PITCH_CANDIDATES,
)


# ---------- File and schema helpers ----------
def list_raw_csvs(folder: str) -> List[str]:
    """Return raw landing CSVs while skipping generated pipeline outputs."""
    all_csvs = glob.glob(os.path.join(folder, "*.csv"))
    exclude = [
        "processed_landings",
        "scenario_metrics",
        "landing_type_performance_stats",
        "permutation_importance_vs",
        "landing_ml_dataset",
        "with_models",
        "dataset",
        "models",
    ]
    return [f for f in all_csvs if not any(s in os.path.basename(f).lower() for s in exclude)]


def scenario_from_name(fname: str, keywords: List[str]) -> str:
    """Infer scenario label from filename keywords."""
    name = fname.lower()
    for k in keywords:
        if k in name:
            return k
    return "unclassified_scenario"


def parse_timestamp_col(df: pd.DataFrame) -> pd.Series:
    """Build a timestamp series from available time columns."""
    if 'realWorldTime' in df.columns:
        ts = pd.to_datetime(df['realWorldTime'], format="%Y%m%d_%H%M%S", errors="coerce")
        if ts.notna().any():
            return ts

    needed = {'hourZ', 'minZ', 'secZ', 'dayZ', 'monthZ', 'yearZ'}
    if needed.issubset(df.columns):
        return pd.to_datetime(
            df[['yearZ', 'monthZ', 'dayZ', 'hourZ', 'minZ', 'secZ']].rename(
                columns={
                    'yearZ': 'year',
                    'monthZ': 'month',
                    'dayZ': 'day',
                    'hourZ': 'hour',
                    'minZ': 'minute',
                    'secZ': 'second'
                }
            ),
            errors="coerce"
        )

    return pd.Series(pd.NaT, index=df.index)


def wrap_deg_to_180(x: float) -> float:
    """Wrap any heading/angle to the [-180, 180) range."""
    return (x + 180.0) % 360.0 - 180.0


def wind_components_true(wind_spd_kt: float, wind_dir_true_deg_from: float, ref_bearing_true_deg: float) -> Tuple[float, float]:
    """Resolve wind into headwind and crosswind components."""
    if pd.isna(wind_spd_kt) or pd.isna(wind_dir_true_deg_from):
        return (np.nan, np.nan)
    rel = math.radians(wrap_deg_to_180(wind_dir_true_deg_from - ref_bearing_true_deg))
    head = wind_spd_kt * math.cos(rel)
    cross = wind_spd_kt * math.sin(rel)
    return head, cross


def coerce_numeric_cols(df: pd.DataFrame, exclude: List[str] = None) -> pd.DataFrame:
    """Coerce non-excluded columns to numeric, invalid values become NaN."""
    exclude = set(exclude or [])
    for c in df.columns:
        if c not in exclude:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df


def choose_first_present(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Pick the first available column from a list of schema candidates."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_throttle_cols(df: pd.DataFrame) -> List[str]:
    """Find throttle-related columns using known names plus keyword fallback."""
    found = []
    for col in THROTTLE_CANDIDATES:
        if col in df.columns:
            found.append(col)
    for c in df.columns:
        if any(s in c.lower() for s in ["thr", "throttle"]) and c not in found:
            found.append(c)
    return found


def is_bank_raw_series(series: pd.Series) -> bool:
    """Heuristic to detect raw encoded bank values versus degree values."""
    if series.dropna().empty:
        return False
    max_abs = series.abs().max()
    if max_abs > 180.0:
        return True
    if pd.api.types.is_integer_dtype(series.dropna()):
        return True
    return False


# ---------- Aircraft/QAR normalization ----------
def apply_qar_conventions(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize controls and derive common attitude columns."""
    if 'aileronIndicator' in df.columns:
        df['aileron_norm'] = df['aileronIndicator'] / CTRL_MAX_ABS
    if 'elevatorIndicator' in df.columns:
        df['elevator_norm'] = df['elevatorIndicator'] / CTRL_MAX_ABS
    if 'rudderIndicator' in df.columns:
        df['rudder_norm'] = df['rudderIndicator'] / CTRL_MAX_ABS

    if 'ias' in df.columns:
        df['ias'] = pd.to_numeric(df['ias'], errors='coerce')

    bank_col = choose_first_present(df, BANK_CANDIDATES)
    if bank_col:
        s = pd.to_numeric(df[bank_col], errors='coerce')
        df['bank_deg'] = s * (360.0 / (65536.0 ** 2)) if is_bank_raw_series(s) else s.astype(float)
    else:
        df['bank_deg'] = np.nan

    pitch_col = choose_first_present(df, PITCH_CANDIDATES)
    if pitch_col:
        df['pitch_deg'] = -pd.to_numeric(df[pitch_col], errors='coerce') if pitch_col == "pit" else pd.to_numeric(df[pitch_col], errors='coerce')
    else:
        df['pitch_deg'] = np.nan

    aoa_col = choose_first_present(df, AOA_CANDIDATES)
    df['aoa_deg'] = pd.to_numeric(df[aoa_col], errors='coerce') if aoa_col else np.nan

    return df


def estimate_dt_seconds(ts: pd.Series) -> float:
    """Estimate sample period from timestamps with a safe fallback."""
    if ts.isna().all():
        return FALLBACK_DT_SECONDS
    diffs = ts.diff().dt.total_seconds()
    med = np.nanmedian(diffs)
    return float(med) if not (np.isnan(med) or med <= 0) else FALLBACK_DT_SECONDS


def construct_alt_agl_fallback(df: pd.DataFrame, td_idx: Optional[int] = None) -> pd.Series:
    """Build AGL altitude from radio alt or corrected baro alt fallback."""
    if 'alt_radio' in df.columns and df['alt_radio'].notna().sum() > 0:
        alt_agl = pd.to_numeric(df['alt_radio'], errors='coerce')
    else:
        baro_col = choose_first_present(df, BARO_ALT_CANDIDATES)
        if baro_col:
            baro = pd.to_numeric(df[baro_col], errors='coerce').copy()
            qnh_correction_ft = (QNH_HPA - 1013.25) * HPA_TO_FEET
            alt_msl = baro - qnh_correction_ft
            alt_agl = alt_msl - RUNWAY_ELEVATION_FT
        else:
            return pd.Series(np.nan, index=df.index)

    if td_idx is not None and 0 <= td_idx < len(alt_agl) and pd.notna(alt_agl.iloc[td_idx]):
        return alt_agl - alt_agl.iloc[td_idx]
    return alt_agl - alt_agl.min(skipna=True)


def find_touchdown_index(df: pd.DataFrame, alt_agl: pd.Series) -> Optional[int]:
    """Detect touchdown from onGround transition, then from min AGL fallback."""
    if 'onGround' in df.columns:
        og = pd.to_numeric(df['onGround'], errors='coerce').fillna(0).astype(int)
        trans = np.where((og.shift(1, fill_value=0) == 0) & (og == 1))[0]
        if len(trans) > 0:
            return int(trans[0])
    if alt_agl.notna().sum() > 0 and alt_agl.min() < 10:
        return int(alt_agl.idxmin())
    return None


def slope_over_window(x: pd.Series, y: pd.Series) -> float:
    """Return linear slope of y over x for a short window."""
    try:
        xv = x.values.astype(float)
        yv = y.values.astype(float)
        if len(xv) < 2 or np.any(np.isnan(xv)) or np.any(np.isnan(yv)):
            return np.nan
        a = np.vstack([xv, np.ones_like(xv)]).T
        m, _ = np.linalg.lstsq(a, yv, rcond=None)[0]
        return float(m)
    except Exception:
        return np.nan


# ---------- Single-file feature extraction ----------
def process_file(fpath: str) -> Optional[Dict[str, Any]]:
    """Extract one landing record worth of engineered features from a CSV."""
    try:
        df = pd.read_csv(fpath, low_memory=False)
    except Exception as e:
        print(f"[WARN] Skipping {fpath}: {e}")
        return None

    fname = os.path.basename(fpath)
    scenario = scenario_from_name(fname, SCENARIO_KEYWORDS)

    df = df.drop_duplicates().reset_index(drop=True)
    required_any = ['onGround', 'vs', 'accVERTy', 'ias']
    if not all(c in df.columns for c in required_any):
        print(f"[WARN] Missing required columns in {fname}")
        return None

    df = coerce_numeric_cols(df, exclude=['realWorldTime', 'scenario'])
    df['timestamp'] = parse_timestamp_col(df)
    df = apply_qar_conventions(df)

    # Compute wind components in runway-aligned coordinates.
    if 'windSpd' in df.columns and 'windDir' in df.columns:
        pairs = df.apply(
            lambda r: wind_components_true(
                r.get('windSpd', np.nan),
                r.get('windDir', np.nan),
                RUNWAY_TRUE_HEADING_DEG
            ),
            axis=1
        )
        df['headwind_kt'] = [p[0] for p in pairs]
        df['crosswind_kt'] = [p[1] for p in pairs]

    alt_agl_initial = construct_alt_agl_fallback(df, td_idx=None)
    td_idx = find_touchdown_index(df, alt_agl_initial)
    if td_idx is None:
        td_idx = find_touchdown_index(df, alt_agl_initial - alt_agl_initial.min(skipna=True))
    if td_idx is None:
        print(f"[WARN] Could not determine touchdown in {fname}")
        return None

    alt_agl = construct_alt_agl_fallback(df, td_idx=td_idx)
    df['alt_agl_ft'] = alt_agl

    # Build the pre-touchdown segment used for approach statistics.
    dt = estimate_dt_seconds(df['timestamp'])
    rows_pre = int(max(1, round(N_SECONDS_BEFORE_TD / dt)))
    idx_pre = max(0, td_idx - rows_pre)
    seg = df.iloc[idx_pre: td_idx + 1].copy()

    if df['timestamp'].isna().all():
        df['_tsec'] = np.arange(len(df)) * dt
        seg['_tsec'] = np.arange(len(seg)) * dt
    else:
        t0 = df['timestamp'].iloc[idx_pre]
        df['_tsec'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
        seg['_tsec'] = (seg['timestamp'] - t0).dt.total_seconds()

    vs_td = df.at[td_idx, 'vs']
    g_td_raw = df.at[td_idx, 'accVERTy']

    # Resolve schema variants for requested raw performance signals.
    aileron_raw_col = choose_first_present(df, ['aileronIndicator', 'aileronindicator'])
    elevator_raw_col = choose_first_present(df, ['elevatorIndicator', 'elevatorindicator'])
    rudder_raw_col = choose_first_present(df, ['rudderIndicator', 'rudderindicator'])
    eng_thr1_col = choose_first_present(df, ['engThrustLever1', 'engthrustlever1', 'eng1ThrLever', 'eng1thrlever'])
    eng_thr2_col = choose_first_present(df, ['engThrustLever2', 'engthrustlever2', 'eng2ThrLever', 'eng2thrlever'])
    wind_spd_col = choose_first_present(df, ['windSpd', 'windspd'])

    feats_at_td = {
        'ias_td': df.at[td_idx, 'ias'],
        'aileronIndicator_td': df.at[td_idx, aileron_raw_col] if aileron_raw_col else np.nan,
        'elevatorIndicator_td': df.at[td_idx, elevator_raw_col] if elevator_raw_col else np.nan,
        'rudderIndicator_td': df.at[td_idx, rudder_raw_col] if rudder_raw_col else np.nan,
        'engThrustLever1_td': df.at[td_idx, eng_thr1_col] if eng_thr1_col else np.nan,
        'engThrustLever2_td': df.at[td_idx, eng_thr2_col] if eng_thr2_col else np.nan,
        'windSpd_td': df.at[td_idx, wind_spd_col] if wind_spd_col else np.nan,
        'bank_td_deg': df.at[td_idx, 'bank_deg'] if 'bank_deg' in df.columns else np.nan,
        'pitch_td_deg': df.at[td_idx, 'pitch_deg'] if 'pitch_deg' in df.columns else np.nan,
        'aoa_td_deg': df.at[td_idx, 'aoa_deg'] if 'aoa_deg' in df.columns else np.nan,
        'alt_td_agl_ft': df.at[td_idx, 'alt_agl_ft'],
        'aileron_td': df.at[td_idx, 'aileron_norm'] if 'aileron_norm' in df.columns else np.nan,
        'elevator_td': df.at[td_idx, 'elevator_norm'] if 'elevator_norm' in df.columns else np.nan,
        'rudder_td': df.at[td_idx, 'rudder_norm'] if 'rudder_norm' in df.columns else np.nan,
        'headwind_td': df.at[td_idx, 'headwind_kt'] if 'headwind_kt' in df.columns else np.nan,
        'crosswind_td': df.at[td_idx, 'crosswind_kt'] if 'crosswind_kt' in df.columns else np.nan,
    }

    def get_seg(name):
        return seg[name] if name in seg.columns else pd.Series(np.nan, index=seg.index)

    seg_vs = get_seg('vs')
    seg_ias = get_seg('ias')
    seg_bank = get_seg('bank_deg')
    seg_pitch = get_seg('pitch_deg')
    seg_aoa = get_seg('aoa_deg')
    seg_alt = get_seg('alt_agl_ft')
    seg_ail = get_seg('aileron_norm')
    seg_ele = get_seg('elevator_norm')
    seg_rud = get_seg('rudder_norm')
    seg_t = seg['_tsec']

    throttle_cols = detect_throttle_cols(seg)
    seg['throttle_mean'] = seg[throttle_cols].mean(axis=1) if throttle_cols else np.nan

    def agg_stats(s: pd.Series, prefix: str) -> Dict[str, float]:
        """Compute a compact set of robust summary stats for one signal."""
        return {
            f'{prefix}_mean': s.mean(skipna=True),
            f'{prefix}_std': s.std(skipna=True),
            f'{prefix}_max': s.max(skipna=True),
            f'{prefix}_min': s.min(skipna=True),
            f'{prefix}_p10': s.quantile(0.10) if s.notna().sum() else np.nan,
            f'{prefix}_p90': s.quantile(0.90) if s.notna().sum() else np.nan,
            f'{prefix}_range': (s.max(skipna=True) - s.min(skipna=True)) if s.notna().sum() else np.nan
        }

    features = {
        'file': fname,
        'scenario': scenario,
        'vs_td_target': vs_td,
        'g_td_target': g_td_raw,
        **feats_at_td,
        'slope_vs': slope_over_window(seg_t, seg_vs),
        'slope_ias': slope_over_window(seg_t, seg_ias),
        'slope_alt': slope_over_window(seg_t, seg_alt),
        'slope_bank': slope_over_window(seg_t, seg_bank),
        'slope_pitch': slope_over_window(seg_t, seg_pitch),
        **agg_stats(seg_vs, 'vs_approach'),
        **agg_stats(seg_ias, 'ias_approach'),
        **agg_stats(seg_bank.abs(), 'bankAbs_approach'),
        **agg_stats(seg_pitch, 'pitch_approach'),
        **agg_stats(seg_aoa, 'aoa_approach'),
        **agg_stats(seg_alt, 'alt_agl_approach'),
        **agg_stats(seg_ail, 'aileron_approach'),
        **agg_stats(seg_ele, 'elevator_approach'),
        **agg_stats(seg_rud, 'rudder_approach'),
        **agg_stats(seg['throttle_mean'], 'throttle_approach')
    }

    # Calculate descent rate from 60 ft AGL to touchdown.
    # Standard gentle descent: -150 ft/sec.
    if 'alt_agl_ft' in df.columns:
        alt_agl_series = pd.to_numeric(df['alt_agl_ft'], errors='coerce')
        pre_td_alts = alt_agl_series.iloc[:td_idx+1]
        # Find the last crossing at or below 60 ft AGL
        below_60 = (pre_td_alts <= 60).values
        if np.any(below_60):
            idx_60ft = np.where(below_60)[0][-1]
        else:
            # If never reaches 60 ft, find closest to 60 ft
            idx_60ft = (pre_td_alts - 60.0).abs().idxmin()
        
        if idx_60ft is not None and idx_60ft < td_idx and '_tsec' in df.columns:
            alt_at_60ft = df.at[idx_60ft, 'alt_agl_ft']
            alt_at_td = df.at[td_idx, 'alt_agl_ft']
            time_elapsed = df.at[td_idx, '_tsec'] - df.at[idx_60ft, '_tsec']
            features['alt_rate_final_60ft'] = (
                (alt_at_td - alt_at_60ft) / time_elapsed
                if time_elapsed > 0 else np.nan
            )
        else:
            features['alt_rate_final_60ft'] = np.nan
    else:
        features['alt_rate_final_60ft'] = np.nan

    features['hard_landing_flag'] = 1 if (pd.notna(vs_td) and vs_td <= HARD_LANDING_FPM_THRESHOLD) else 0
    return features


# ---------- Dataset assembly ----------
def load_and_engineer_dataset() -> pd.DataFrame:
    """Process all raw files into one engineered modeling dataset."""
    files = list_raw_csvs(DATA_FOLDER)
    if not files:
        print(f"No CSV files found in {DATA_FOLDER}")
        return pd.DataFrame()

    records = [process_file(f) for f in files]
    df_all = pd.DataFrame([r for r in records if r is not None])
    if df_all.empty:
        print("No valid landing data found.")
        return df_all

    # Filter out landings with VS = 0 at touchdown (sensor didn't register descent).
    vs_zero = df_all['vs_td_target'] == 0
    if vs_zero.any():
        bad_files = df_all.loc[vs_zero, 'file'].tolist()
        print(f"[INFO] Dropping {vs_zero.sum()} landing(s) with VS=0 at touchdown "
              f"(sensor error): {bad_files}")
        df_all = df_all[~vs_zero].reset_index(drop=True)

    # Flag unrealistic G values (outside ±5g) as NaN so they are excluded from
    # G-model training but the row is still used for the VS model.
    G_REALISTIC_LIMIT = 5.0
    g_raw = pd.to_numeric(df_all['g_td_target'], errors='coerce')
    bad_g = g_raw.abs() > G_REALISTIC_LIMIT
    if bad_g.any():
        bad_files = df_all.loc[bad_g, 'file'].tolist()
        print(f"[INFO] {bad_g.sum()} landing(s) have unrealistic G (|g|>{G_REALISTIC_LIMIT}); "
              f"G target set to NaN: {bad_files}")
        g_raw[bad_g] = np.nan

    df_all['g_td_target_clipped'] = g_raw.clip(lower=-3, upper=3)
    df_all['g_td_target_smooth'] = (
        df_all['g_td_target_clipped'].rolling(window=3, center=True, min_periods=1).mean()
    )
    df_all['g_td_target'] = df_all['g_td_target_smooth']

    # Save intermediate engineered features for quick inspection/debug.
    out_raw = os.path.join(OUTPUT_FOLDER, "processed_landings_raw.csv")
    df_all.to_csv(out_raw, index=False)
    print(f"[INFO] Saved raw engineered features to {out_raw}")
    return df_all


def extract_control_timeseries(file_path: str, seconds_before_td: float = 45.0) -> pd.DataFrame:
    """Extract per-sample control traces in the final N seconds before touchdown."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"[WARN] Skipping control trace for {file_path}: {e}")
        return pd.DataFrame()

    file_name = os.path.basename(file_path)
    scenario = scenario_from_name(file_name, SCENARIO_KEYWORDS)

    df = df.drop_duplicates().reset_index(drop=True)
    required_any = ['onGround', 'vs', 'accVERTy', 'ias']
    if not all(c in df.columns for c in required_any):
        return pd.DataFrame()

    df = coerce_numeric_cols(df, exclude=['realWorldTime', 'scenario'])
    df['timestamp'] = parse_timestamp_col(df)
    df = apply_qar_conventions(df)

    alt_agl_initial = construct_alt_agl_fallback(df, td_idx=None)
    td_idx = find_touchdown_index(df, alt_agl_initial)
    if td_idx is None:
        td_idx = find_touchdown_index(df, alt_agl_initial - alt_agl_initial.min(skipna=True))
    if td_idx is None:
        return pd.DataFrame()

    dt = estimate_dt_seconds(df['timestamp'])
    rows_pre = int(max(1, round(seconds_before_td / dt)))
    idx_pre = max(0, td_idx - rows_pre)
    seg = df.iloc[idx_pre: td_idx + 1].copy()

    if df['timestamp'].isna().all():
        seg['_tsec'] = np.arange(len(seg)) * dt
    else:
        t0 = seg['timestamp'].iloc[0]
        seg['_tsec'] = (seg['timestamp'] - t0).dt.total_seconds()

    td_t = seg['_tsec'].iloc[-1]
    seg['time_to_td_s'] = seg['_tsec'] - td_t

    for col in ['aileron_norm', 'elevator_norm', 'rudder_norm']:
        if col not in seg.columns:
            seg[col] = np.nan

    out = seg[['time_to_td_s', 'aileron_norm', 'elevator_norm', 'rudder_norm']].copy()
    out['scenario'] = scenario
    out['file'] = file_name
    out = out[(out['time_to_td_s'] >= -seconds_before_td) & (out['time_to_td_s'] <= 0)]
    return out


def build_control_timeseries_dataset(seconds_before_td: float = 45.0) -> pd.DataFrame:
    """Build stacked control-input traces for all files, labeled by scenario."""
    files = list_raw_csvs(DATA_FOLDER)
    if not files:
        return pd.DataFrame()

    parts = [extract_control_timeseries(f, seconds_before_td=seconds_before_td) for f in files]
    df_ctrl = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    return df_ctrl


def extract_control_profile_by_agl(file_path: str, max_agl_ft: float = 600.0) -> pd.DataFrame:
    """Extract control inputs from max_agl_ft down to touchdown (0 ft AGL)."""
    try:
        df = pd.read_csv(file_path, low_memory=False)
    except Exception as e:
        print(f"[WARN] Skipping AGL profile for {file_path}: {e}")
        return pd.DataFrame()

    file_name = os.path.basename(file_path)
    scenario = scenario_from_name(file_name, SCENARIO_KEYWORDS)

    df = df.drop_duplicates().reset_index(drop=True)
    required_any = ['onGround', 'vs', 'accVERTy', 'ias']
    if not all(c in df.columns for c in required_any):
        return pd.DataFrame()

    df = coerce_numeric_cols(df, exclude=['realWorldTime', 'scenario'])
    df['timestamp'] = parse_timestamp_col(df)
    df = apply_qar_conventions(df)

    alt_agl_initial = construct_alt_agl_fallback(df, td_idx=None)
    td_idx = find_touchdown_index(df, alt_agl_initial)
    if td_idx is None:
        td_idx = find_touchdown_index(df, alt_agl_initial - alt_agl_initial.min(skipna=True))
    if td_idx is None:
        return pd.DataFrame()

    # Normalize AGL so touchdown is near 0 ft.
    df['alt_agl_ft'] = construct_alt_agl_fallback(df, td_idx=td_idx)

    # Keep the final approach segment from first crossing below max_agl_ft to touchdown.
    pre_td = df.iloc[:td_idx + 1].copy()
    above_idx = np.where(pd.to_numeric(pre_td['alt_agl_ft'], errors='coerce').fillna(-1) > max_agl_ft)[0]
    start_idx = int(above_idx[-1] + 1) if len(above_idx) > 0 else 0
    seg = pre_td.iloc[start_idx: td_idx + 1].copy()

    for col in ['aileron_norm', 'elevator_norm', 'rudder_norm']:
        if col not in seg.columns:
            seg[col] = np.nan

    seg['alt_agl_ft'] = pd.to_numeric(seg['alt_agl_ft'], errors='coerce')
    out = seg[['alt_agl_ft', 'aileron_norm', 'elevator_norm', 'rudder_norm']].copy()
    out = out[(out['alt_agl_ft'] >= 0) & (out['alt_agl_ft'] <= max_agl_ft)]
    if out.empty:
        return out

    out['agl_bin_ft'] = out['alt_agl_ft'].round().astype(int)
    out['scenario'] = scenario
    out['file'] = file_name
    return out


def build_control_agl_dataset(max_agl_ft: float = 600.0) -> pd.DataFrame:
    """Build stacked control profiles by AGL bin for all files."""
    files = list_raw_csvs(DATA_FOLDER)
    if not files:
        return pd.DataFrame()

    parts = [extract_control_profile_by_agl(f, max_agl_ft=max_agl_ft) for f in files]
    df_ctrl_agl = pd.concat([p for p in parts if not p.empty], ignore_index=True) if parts else pd.DataFrame()
    return df_ctrl_agl
