from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from model_io import load_model

DATA_DIR = Path(__file__).resolve().parent / "data"
DATASET_PATH = DATA_DIR / "merged_dataset.csv"
MODEL_PATH = DATA_DIR / "model.joblib"


def create_visualizations(data_path: Path = DATASET_PATH, model_path: Path = MODEL_PATH) -> None:
    df = pd.read_csv(data_path)
    model = load_model(model_path)

    df_numeric = df.select_dtypes(include=["number"]).copy()

    sns.set_theme(style="whitegrid")

    # 1) Correlation heatmap of numeric features
    plt.figure(figsize=(10, 8))
    corr = df_numeric.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "correlation_heatmap.png", dpi=300)
    plt.close()

    # 2) Scatter plot: tempo vs productivity
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x="tempo", y="productivity", alpha=0.7)
    plt.title("Tempo vs Productivity")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "tempo_vs_productivity_scatter.png", dpi=300)
    plt.close()

    # 3) Boxplot of productivity grouped by binned cognitive_load_final
    binned = df.copy()
    binned["cognitive_load_bin"] = pd.cut(
        binned["cognitive_load_final"], bins=4, labels=["Low", "Medium", "High", "Very High"]
    )
    plt.figure(figsize=(9, 6))
    sns.boxplot(data=binned, x="cognitive_load_bin", y="productivity")
    plt.title("Productivity by Cognitive Load Bin")
    plt.xlabel("Cognitive Load (Binned)")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "productivity_by_cognitive_load_boxplot.png", dpi=300)
    plt.close()

    # 4) Bar chart of Random Forest feature importances
    if not hasattr(model, "feature_importances_"):
        raise ValueError("Loaded model does not expose feature_importances_ (expected Random Forest model)")

    feature_names = ["tempo", "energy", "valence", "cognitive_load_final"]
    if len(model.feature_importances_) != len(feature_names):
        feature_names = [f"feature_{i}" for i in range(len(model.feature_importances_))]

    importance_df = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    plt.figure(figsize=(8, 6))
    sns.barplot(data=importance_df, x="feature", y="importance", palette="viridis")
    plt.title("Random Forest Feature Importances")
    plt.ylabel("Importance")
    plt.xlabel("Feature")
    plt.tight_layout()
    plt.savefig(DATA_DIR / "rf_feature_importances.png", dpi=300)
    plt.close()

    print("Saved visualizations:")
    print(f"- {DATA_DIR / 'correlation_heatmap.png'}")
    print(f"- {DATA_DIR / 'tempo_vs_productivity_scatter.png'}")
    print(f"- {DATA_DIR / 'productivity_by_cognitive_load_boxplot.png'}")
    print(f"- {DATA_DIR / 'rf_feature_importances.png'}")


if __name__ == "__main__":
    create_visualizations()
