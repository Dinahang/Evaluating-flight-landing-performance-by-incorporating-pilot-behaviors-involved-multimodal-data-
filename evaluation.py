import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------- Model explainability ----------
def global_permutation_report(model, x_test, y_test, feat_names, title: str, n_repeats: int = 3):
    """Plot and return mean permutation importance values."""
    result = permutation_importance(model, x_test, y_test, n_repeats=n_repeats, random_state=42, n_jobs=-1)
    importances = pd.Series(result.importances_mean, index=feat_names).sort_values()
    top = importances.tail(10)

    # Print a readable table so results are visible even if plots are not shown.
    top_table = top.sort_values(ascending=False).reset_index()
    top_table.columns = ['feature', 'importance_mean']
    print("\n[PERMUTATION IMPORTANCE - TOP 10]")
    print(top_table.to_string(index=False))

    plt.figure(figsize=(9, 5))
    top.plot(kind='barh')
    plt.title(title + " (Top 10)")
    plt.tight_layout()
    plt.show()
    return importances


# ---------- Console metrics ----------
def print_global_model_metrics(y_test_vs, preds_vs):
    """Print headline hold-out metrics for VS model."""
    print("\n[GLOBAL VS MODEL]")
    print(f"R^2:  {r2_score(y_test_vs, preds_vs):.4f}")
    print(f"MAE:  {mean_absolute_error(y_test_vs, preds_vs):.2f} fpm")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test_vs, preds_vs)):.2f} fpm")


# ---------- EDA plots ----------
def plot_eda(df_all, y_test_vs, preds_vs):
    """Render diagnostic plots for VS prediction quality and target distributions."""
    try:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        ax.scatter(y_test_vs, preds_vs, alpha=0.6, edgecolor='k')
        lims = [min(y_test_vs.min(), preds_vs.min()), max(y_test_vs.max(), preds_vs.max())]
        ax.plot(lims, lims, 'r--')
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_title("VS: Predicted vs Actual")
        ax.set_xlabel("Actual VS (fpm)")
        ax.set_ylabel("Predicted VS (fpm)")
        plt.tight_layout()
        plt.show()

        err_vs = preds_vs - y_test_vs.values
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        sns.histplot(err_vs, bins=30, kde=True, ax=ax, color='tab:blue')
        ax.set_title("VS Residuals (Pred - Actual) [fpm]")
        plt.tight_layout()
        plt.show()

        if 'scenario' in df_all.columns:
            fig, ax = plt.subplots(1, 1, figsize=(9, 5))
            sns.boxplot(data=df_all, x='scenario', y='vs_td_target', ax=ax)
            ax.set_title("VS by Scenario (Touchdown fpm)")
            ax.tick_params(axis='x', rotation=30)
            plt.tight_layout()
            plt.show()

        fig, ax = plt.subplots(1, 1, figsize=(7, 4))
        sns.histplot(df_all, x='vs_td_target', hue='hard_landing_flag', kde=True, ax=ax)
        ax.set_title("Vertical Speed at Touchdown")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print(f"[WARN] EDA plotting skipped due to: {e}")


def plot_control_inputs_by_scenario(df_ctrl: pd.DataFrame, seconds_before_td: float = 45.0):
    """Mean + std errorbar plot matching reference style (time-based)."""
    if df_ctrl is None or df_ctrl.empty:
        print("[WARN] Control input plotting skipped: no control trace data available.")
        return

    plt.style.use('default')

    work = df_ctrl.copy()
    work['time_bin_s'] = work['time_to_td_s'].round().astype(int)
    work = work[(work['time_bin_s'] >= -int(seconds_before_td)) & (work['time_bin_s'] <= 0)]

    scenario_order = sorted(work['scenario'].dropna().unique().tolist())
    controls = [
        ('aileron_norm', 'Aileron (norm)', 'Aileron', '#1f77b4', 'o'),
        ('elevator_norm', 'Elevator (norm)', 'Elevator', '#2ca02c', 's'),
        ('rudder_norm', 'Rudder (norm)', 'Rudder', '#ff7f0e', 'D'),
    ]

    for scen in scenario_order:
        sdata = work[work['scenario'] == scen]
        if sdata.empty:
            continue

        n_flights = sdata['file'].nunique()

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(
            f"Control Surface vs Time - {scen.capitalize()} Scenario (N={n_flights})",
            fontsize=13, fontweight='bold',
        )

        for ax, (col, ylabel, legend_label, color, marker) in zip(axes, controls):
            agg = (
                sdata.groupby('time_bin_s')[col]
                .agg(['mean', 'std'])
                .reset_index()
                .sort_values('time_bin_s')
            )
            agg['std'] = agg['std'].fillna(0)

            ax.errorbar(
                -agg['time_bin_s'], agg['mean'], yerr=agg['std'],
                fmt=f'-{marker}', color=color, ecolor=color,
                capsize=3, capthick=1, elinewidth=1,
                markersize=5, linewidth=1.2, label=legend_label,
            )
            ax.axhline(0.0, color='grey', linewidth=0.8, linestyle='--')
            ax.set_ylim(-1.0, 1.0)
            ax.set_ylabel(ylabel)
            ax.legend(loc='upper right', frameon=True)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Time Before Touchdown (s)')
        axes[-1].set_xticks(np.arange(0, int(seconds_before_td) + 1, 5))
        plt.tight_layout()
        plt.show()


def plot_landing_type_performance(df_all: pd.DataFrame, output_dir: str = None):
    """Plot hard vs soft landing distributions for requested touchdown signals."""
    signals = [
        'aileron_td',
        'elevator_td',
        'rudder_td',
        'engThrustLever1_td',
        'engThrustLever2_td',
        'windSpd_td',
    ]

    friendly_labels = {
        'aileron_td':           ('Aileron (Wing Tilt)',        '% of Full Deflection'),
        'elevator_td':          ('Elevator (Nose Up/Down)',    '% of Full Deflection'),
        'rudder_td':            ('Rudder (Nose Left/Right)',   '% of Full Deflection'),
        'engThrustLever1_td':   ('Engine 1 Thrust Lever',     'Raw QAR Value'),
        'engThrustLever2_td':   ('Engine 2 Thrust Lever',     'Raw QAR Value'),
        'windSpd_td':           ('Wind Speed',                 'Speed (kt)'),
    }

    # Direction annotations for control surfaces (standard aviation convention)
    direction_hints = {
        'aileron_td':  ('+  Right wing down', '\u2212  Left wing down'),
        'elevator_td': ('+  Nose up',         '\u2212  Nose down'),
        'rudder_td':   ('+  Nose right',      '\u2212  Nose left'),
    }

    # Columns that should be displayed as percentage (-100% to +100%)
    pct_cols = {'aileron_td', 'elevator_td', 'rudder_td'}

    work = df_all.copy()
    if 'hard_landing_flag' not in work.columns:
        print("[WARN] Landing type plot skipped: hard_landing_flag is missing.")
        return

    work['landing_type'] = np.where(work['hard_landing_flag'].astype(int) == 1, 'hard', 'soft')

    # Convert normalized (-1 to 1) controls to percentage for readability
    for c in pct_cols:
        if c in work.columns:
            work[c] = pd.to_numeric(work[c], errors='coerce') * 100.0

    available = [c for c in signals if c in work.columns]
    if not available:
        print("[WARN] Landing type plot skipped: required touchdown metrics not found.")
        return

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    fig.suptitle('Landing-Type Performance (Hard vs Soft)')

    for i, col in enumerate(signals):
        ax = axes[i]
        title, ylabel = friendly_labels[col]
        if col not in work.columns or pd.to_numeric(work[col], errors='coerce').notna().sum() == 0:
            ax.set_title(f"{title}\n(no data)")
            ax.axis('off')
            continue

        sns.boxplot(data=work, x='landing_type', y=col, order=['soft', 'hard'], ax=ax)
        sns.stripplot(
            data=work,
            x='landing_type',
            y=col,
            order=['soft', 'hard'],
            alpha=0.35,
            size=3,
            color='black',
            ax=ax,
        )
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Landing Type')

        # Add direction labels for control surface subplots
        if col in direction_hints:
            pos_label, neg_label = direction_hints[col]
            ylims = ax.get_ylim()
            ax.text(1.0, 0.97, pos_label, transform=ax.transAxes,
                    fontsize=8, color='green', ha='right', va='top',
                    fontstyle='italic')
            ax.text(1.0, 0.03, neg_label, transform=ax.transAxes,
                    fontsize=8, color='red', ha='right', va='bottom',
                    fontstyle='italic')
            # Add a faint zero reference line
            ax.axhline(0, color='grey', linewidth=0.7, linestyle='--', alpha=0.5)

    plt.tight_layout()

    if output_dir:
        out_plot = os.path.join(output_dir, 'landing_type_performance.png')
        plt.savefig(out_plot, dpi=180, bbox_inches='tight')
        print(f"[INFO] Landing type plot saved to {out_plot}")

    plt.show()


def plot_control_inputs_by_agl(df_ctrl_agl: pd.DataFrame, max_agl_ft: float = 600.0, output_dir: str = None):
    """Mean + std errorbar plot matching reference style (AGL-based)."""
    if df_ctrl_agl is None or df_ctrl_agl.empty:
        print("[WARN] AGL control-input plotting skipped: no data available.")
        return

    plt.style.use('default')

    work = df_ctrl_agl.copy()
    work = work[(work['agl_bin_ft'] >= 0) & (work['agl_bin_ft'] <= int(max_agl_ft))]
    scenario_order = sorted(work['scenario'].dropna().unique().tolist())
    controls = [
        ('aileron_norm', 'Aileron (norm)', 'Aileron', '#1f77b4', 'o'),
        ('elevator_norm', 'Elevator (norm)', 'Elevator', '#2ca02c', 's'),
        ('rudder_norm', 'Rudder (norm)', 'Rudder', '#ff7f0e', 'D'),
    ]

    for scen in scenario_order:
        sdata = work[work['scenario'] == scen]
        if sdata.empty:
            continue

        n_flights = sdata['file'].nunique()

        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        fig.suptitle(
            f"Control Surface vs Altitude - {scen.capitalize()} Scenario (N={n_flights})",
            fontsize=13, fontweight='bold',
        )

        for ax, (col, ylabel, legend_label, color, marker) in zip(axes, controls):
            agg = (
                sdata.groupby('agl_bin_ft')[col]
                .agg(['mean', 'std'])
                .reset_index()
                .sort_values('agl_bin_ft')
            )
            agg['std'] = agg['std'].fillna(0)

            ax.errorbar(
                agg['agl_bin_ft'], agg['mean'], yerr=agg['std'],
                fmt=f'-{marker}', color=color, ecolor=color,
                capsize=3, capthick=1, elinewidth=1,
                markersize=5, linewidth=1.2, label=legend_label,
            )
            ax.axhline(0.0, color='grey', linewidth=0.8, linestyle='--')
            ax.set_ylim(-1.0, 1.0)
            ax.set_ylabel(ylabel)
            ax.legend(loc='upper right', frameon=True)
            ax.grid(True, alpha=0.3)

        axes[-1].set_xlabel('Altitude AGL (ft)')
        axes[-1].set_xticks(np.arange(0, int(max_agl_ft) + 1, 100))
        plt.tight_layout()

        if output_dir:
            safe_scen = str(scen).replace(' ', '_')
            out_plot = os.path.join(output_dir, f'control_inputs_agl_{int(max_agl_ft)}ft_{safe_scen}.png')
            plt.savefig(out_plot, dpi=180, bbox_inches='tight')
            print(f"[INFO] AGL control plot saved to {out_plot}")

        plt.show()
