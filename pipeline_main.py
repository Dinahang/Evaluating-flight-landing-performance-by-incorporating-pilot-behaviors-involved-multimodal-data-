import os

os.environ['LANG'] = 'en_US.UTF-8'
os.environ['LC_ALL'] = 'en_US.UTF-8'

from evaluation import (  # type: ignore
    global_permutation_report,
    plot_control_inputs_by_agl,
    plot_control_inputs_by_scenario,
    plot_eda,
    plot_landing_type_performance,
    print_global_model_metrics,
)
from feature_engineering import build_control_agl_dataset, build_control_timeseries_dataset, load_and_engineer_dataset  # type: ignore
from modeling import (  # type: ignore
    add_anomaly_labels,
    build_feature_inputs,
    run_hard_landing_classifier,
    save_landing_type_performance_stats,
    save_scenario_metrics,
    train_vs_model,
)
from pipeline_config import DATA_FOLDER, OUTPUT_FOLDER, DO_PERM_IMPORTANCE, PERM_N_REPEATS, configure_runtime  # type: ignore


def save_final_dataset(df_all, x_df):
    """Save final model-ready dataset with targets and metadata columns."""
    out_final = os.path.join(OUTPUT_FOLDER, 'processed_landings.csv')
    x_out = x_df.copy()
    x_out['vs_td_target'] = df_all['vs_td_target'].values
    x_out['hard_landing_flag'] = df_all['hard_landing_flag'].values
    x_out['scenario'] = df_all['scenario'].values
    x_out['file'] = df_all['file'].values
    x_out.to_csv(out_final, index=False)
    print(f"[INFO] Pipeline Complete. Data saved to {out_final}")


def main():
    """Run the full end-to-end landing analytics pipeline."""
    # Runtime display settings and warning filters.
    configure_runtime()

    # Stage 1: load raw CSVs and engineer one row per landing.
    df_all = load_and_engineer_dataset()
    if df_all.empty:
        return

    target_vs = 'vs_td_target'

    # Stage 2: build model inputs.
    x_df, x_filled, y_vs, scen_series, imputer, feature_cols = build_feature_inputs(df_all)

    # Stage 3: train VS regression model with cross-validation.
    model_vs, y_test_vs, preds_vs, x_test_vs_filled = train_vs_model(x_df, y_vs, scen_series)

    print_global_model_metrics(y_test_vs, preds_vs)

    # Stage 4: optional anomaly tags and feature-importance report.
    df_all = add_anomaly_labels(df_all, x_filled)

    if DO_PERM_IMPORTANCE:
        try:
            importances = global_permutation_report(
                model_vs,
                x_test_vs_filled,
                y_test_vs,
                feature_cols,
                'Global Permutation Importance (VS)',
                n_repeats=PERM_N_REPEATS,
            )
            perm_df = importances.sort_values(ascending=False).reset_index()
            perm_df.columns = ['feature', 'importance_mean']
            out_perm = os.path.join(OUTPUT_FOLDER, 'permutation_importance_vs.csv')
            out_perm_top = os.path.join(OUTPUT_FOLDER, 'permutation_importance_vs_top10.csv')
            perm_df.to_csv(out_perm, index=False)
            perm_df.head(10).to_csv(out_perm_top, index=False)
            print(f"[INFO] Permutation importance saved to {out_perm}")
            print(f"[INFO] Top-10 permutation importance saved to {out_perm_top}")
        except Exception as e:
            print(f"[WARN] Permutation importance skipped: {e}")

    # Stage 5: grouped metrics, classifier diagnostics, and plots.
    save_scenario_metrics(df_all, x_df, imputer, model_vs, target_vs)
    run_hard_landing_classifier(df_all, x_filled)
    save_landing_type_performance_stats(df_all)
    plot_landing_type_performance(df_all, output_dir=OUTPUT_FOLDER)
    plot_eda(df_all, y_test_vs, preds_vs)

    # Stage 6: persist final dataset artifact.
    save_final_dataset(df_all, x_df)

    # Stage 7: per-scenario control traces in final 45 seconds before touchdown.
    try:
        df_ctrl = build_control_timeseries_dataset(seconds_before_td=45.0)
        plot_control_inputs_by_scenario(df_ctrl, seconds_before_td=45.0)
    except Exception as e:
        print(f"[WARN] Control input plotting skipped due to: {e}")

    # Stage 8: per-scenario control traces from 600 ft AGL to touchdown.
    try:
        df_ctrl_agl = build_control_agl_dataset(max_agl_ft=600.0)
        plot_control_inputs_by_agl(df_ctrl_agl, max_agl_ft=600.0, output_dir=OUTPUT_FOLDER)
    except Exception as e:
        print(f"[WARN] AGL control-input plotting skipped due to: {e}")


if __name__ == '__main__':
    main()
