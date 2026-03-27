import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split, KFold, cross_val_score

from pipeline_config import OUTPUT_FOLDER  # type: ignore


# ---------- Feature matrix preparation ----------
def build_feature_inputs(df_all: pd.DataFrame):
    """Create model-ready matrix, VS target, scenario labels, and fitted imputer."""
    target_vs = 'vs_td_target'
    base_non_features = ['file', 'scenario', target_vs, 'g_td_target', 'hard_landing_flag']

    base_feature_cols = [c for c in df_all.columns if c not in base_non_features]
    scen_dummies = pd.get_dummies(df_all['scenario'], prefix='scen')
    x_df = pd.concat([df_all[base_feature_cols], scen_dummies], axis=1)

    y_vs = df_all[target_vs].astype(float).copy()
    scen_series = df_all['scenario']

    # Fit imputer on full data just for anomaly detection / final export;
    # the VS model fits its own imputer on training data only (no leakage).
    imputer = SimpleImputer(strategy='median', keep_empty_features=True)
    x_filled = imputer.fit_transform(x_df)
    if hasattr(imputer, 'get_feature_names_out'):
        feature_cols = list(imputer.get_feature_names_out(x_df.columns))
    else:
        feature_cols = list(x_df.columns)

    return x_df, x_filled, y_vs, scen_series, imputer, feature_cols


# ---------- VS Regression model ----------
def train_vs_model(x_df, y_vs, scen_series):
    """Train VS regressor with proper imputation (no leakage) and cross-validation."""
    x_raw = x_df.values if hasattr(x_df, 'values') else x_df

    # --- Hold-out split (stratified by scenario) ---
    idx_all = np.arange(len(x_raw))
    idx_train, idx_test, y_train, y_test, scen_train, scen_test = train_test_split(
        idx_all, y_vs, scen_series, test_size=0.2, random_state=42, stratify=scen_series
    )

    # Fit imputer on TRAINING data only (fixes data leakage).
    imputer_vs = SimpleImputer(strategy='median', keep_empty_features=True)
    x_train = imputer_vs.fit_transform(x_raw[idx_train])
    x_test = imputer_vs.transform(x_raw[idx_test])

    # --- Cross-validation on training set to pick best model ---
    candidates = {
        'RF_conservative': RandomForestRegressor(
            n_estimators=200, max_depth=5, min_samples_leaf=5,
            max_features=0.5, random_state=45, n_jobs=-1,
        ),
        'RF_moderate': RandomForestRegressor(
            n_estimators=300,
            max_depth=8,
            min_samples_leaf=3,
            max_features='sqrt', random_state=45, n_jobs=-1,
        ),
        'GBR': GradientBoostingRegressor(
            n_estimators=200, max_depth=4, min_samples_leaf=4,
            learning_rate=0.05,
            subsample=0.8, random_state=45,
        ),
    }

    n_samples = len(x_train)
    cv_strategy = KFold(n_splits=min(5, n_samples), shuffle=True, random_state=42)

    print("\n[VS MODEL SELECTION \u2014 Cross-Validation]")
    best_name, best_score = None, np.inf
    best_model = None
    for name, model in candidates.items():
        scores = cross_val_score(model, x_train, y_train, cv=cv_strategy,
                                 scoring='neg_mean_absolute_error')
        mean_mae = -scores.mean()
        std_mae = scores.std()
        print(f"  {name:20s}  CV MAE: {mean_mae:.2f} \u00b1 {std_mae:.2f} fpm")
        if mean_mae < best_score:
            best_name, best_score, best_model = name, mean_mae, model

    print(f"  >> Best: {best_name} (CV MAE = {best_score:.2f} fpm)")

    # --- Retrain best model on full training set, evaluate on hold-out ---
    assert best_model is not None, "No candidate model was selected"
    best_model.fit(x_train, y_train)
    preds = best_model.predict(x_test)

    return best_model, y_test, preds, x_test


def add_anomaly_labels(df_all: pd.DataFrame, x_filled) -> pd.DataFrame:
    """Tag unusual samples for downstream inspection/EDA."""
    iso = IsolationForest(contamination=0.10, random_state=42)
    df_all['anomaly_label'] = iso.fit_predict(x_filled)
    return df_all


# ---------- Grouped evaluation artifacts ----------
def save_scenario_metrics(df_all, x_df, imputer, model_vs, target_vs: str):
    """Compute and save per-scenario VS regression metrics."""
    scenario_metrics = []
    for scen_name, df_s in df_all.groupby('scenario'):
        if len(df_s) < 5:
            continue

        xs = x_df.loc[df_s.index]
        xs_filled = imputer.transform(xs)
        ys_vs = df_s[target_vs].values

        p_vs = model_vs.predict(xs_filled)

        scenario_metrics.append(
            {
                'scenario': scen_name,
                'n': len(df_s),
                'VS_R2': r2_score(ys_vs, p_vs) if len(np.unique(ys_vs)) > 1 else np.nan,
                'VS_MAE_fpm': mean_absolute_error(ys_vs, p_vs),
                'VS_RMSE_fpm': np.sqrt(mean_squared_error(ys_vs, p_vs)),
            }
        )

    if scenario_metrics:
        df_scen = pd.DataFrame(scenario_metrics)
        out_scen = os.path.join(OUTPUT_FOLDER, 'scenario_metrics.csv')
        df_scen.to_csv(out_scen, index=False)
        print(f"[INFO] Per-scenario metrics saved to {out_scen}")
        print(df_scen.sort_values('scenario'))


# ---------- Classification helper ----------
def run_hard_landing_classifier(df_all, x_filled):
    """Train and report a simple hard-landing classifier on hold-out data."""
    print("\n[SIMPLE HARD-LANDING CLASSIFIER (NO SMOTE)]")
    y_cls = df_all['hard_landing_flag'].astype(int).values

    x_train_c, x_test_c, y_train_c, y_test_c = train_test_split(
        x_filled, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(x_train_c, y_train_c)
    pred_c = clf.predict(x_test_c)
    try:
        proba_c = clf.predict_proba(x_test_c)[:, 1]
    except Exception:
        proba_c = None

    print("Confusion matrix (hold-out):\n", confusion_matrix(y_test_c, pred_c))
    print("\nClassification report (hold-out):\n", classification_report(y_test_c, pred_c, digits=3))
    if proba_c is not None:
        try:
            print("ROC-AUC (hold-out): %.3f" % roc_auc_score(y_test_c, proba_c))
        except Exception:
            pass


def save_landing_type_performance_stats(df_all: pd.DataFrame):
    """Summarize key touchdown performance signals by landing type (hard vs soft)."""
    signals = [
        'aileronIndicator_td',
        'elevatorIndicator_td',
        'rudderIndicator_td',
        'engThrustLever1_td',
        'engThrustLever2_td',
        'windSpd_td',
    ]

    work = df_all.copy()
    work['landing_type'] = np.where(work['hard_landing_flag'].astype(int) == 1, 'hard', 'soft')

    rows = []
    for landing_type in ['hard', 'soft']:
        grp = work[work['landing_type'] == landing_type]
        for col in signals:
            if col in grp.columns:
                s = pd.to_numeric(grp[col], errors='coerce').dropna()
            else:
                s = pd.Series(dtype=float)

            if s.empty:
                rows.append(
                    {
                        'landing_type': landing_type,
                        'metric': col,
                        'n': 0,
                        'mean': np.nan,
                        'std': np.nan,
                        'median': np.nan,
                        'p10': np.nan,
                        'p90': np.nan,
                        'min': np.nan,
                        'max': np.nan,
                    }
                )
            else:
                rows.append(
                    {
                        'landing_type': landing_type,
                        'metric': col,
                        'n': int(s.shape[0]),
                        'mean': float(s.mean()),
                        'std': float(s.std(ddof=1)) if s.shape[0] > 1 else np.nan,
                        'median': float(s.median()),
                        'p10': float(s.quantile(0.10)),
                        'p90': float(s.quantile(0.90)),
                        'min': float(s.min()),
                        'max': float(s.max()),
                    }
                )

    if not rows:
        print("[WARN] Landing type stats skipped: no non-null values in requested signals.")
        return

    stats_df = pd.DataFrame(rows)
    out_stats = os.path.join(OUTPUT_FOLDER, 'landing_type_performance_stats.csv')
    stats_df.to_csv(out_stats, index=False)
    print(f"\n[LANDING TYPE PERFORMANCE STATS]")
    print(stats_df.sort_values(['metric', 'landing_type']).to_string(index=False))
    print(f"[INFO] Landing type stats saved to {out_stats}")
