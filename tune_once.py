from feature_engineering import load_and_engineer_dataset
from modeling import build_feature_inputs, train_vs_model


def main() -> None:
    print("[RUN] Loading and engineering dataset...")
    df_all = load_and_engineer_dataset()
    if df_all.empty:
        print("[RUN] No data available.")
        return

    x_df, x_filled, y_vs, scen_series, imputer, feature_cols = build_feature_inputs(df_all)
    print("[RUN] Starting VS model tuning only...")
    train_vs_model(x_df, y_vs, scen_series)
    print("[RUN] Done.")


if __name__ == "__main__":
    main()
