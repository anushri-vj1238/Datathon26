from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"

SPOTIFY_COLUMNS = [
    "tempo",
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "loudness",
]

EMOTION_MAP = {"happy": 0, "neutral": 1, "sad": 2}
COGNITIVE_WEIGHTS = {
    "emotion_score": 0.4,
    "stress_index": 0.3,
    "trait_load": 0.2,
    "EEG_feature": 0.1,
}


def scale_1_to_10(series: pd.Series) -> pd.Series:
    """Min-max normalize a numeric series to the range [1, 10]."""
    s = pd.to_numeric(series, errors="coerce").astype(float)
    min_val, max_val = s.min(), s.max()
    if pd.isna(min_val) or pd.isna(max_val):
        return pd.Series(np.nan, index=s.index, dtype=float)
    if np.isclose(max_val, min_val):
        return pd.Series(5.5, index=s.index, dtype=float)
    return 1.0 + 9.0 * (s - min_val) / (max_val - min_val)


def normalize_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    out = df.copy()
    for col in columns:
        out[col] = scale_1_to_10(out[col])
    return out


def preprocess_spotify(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "spotify.csv")
    available = [c for c in SPOTIFY_COLUMNS if c in df.columns]
    if not available:
        raise ValueError("spotify.csv is missing all required Spotify feature columns")

    out = df[available].copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype(float)
    out = out.dropna().reset_index(drop=True)
    return out


def preprocess_emotion(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "emotion.csv")

    label_col = None
    for candidate in ["emotion", "emotion_label", "label", "mood"]:
        if candidate in df.columns:
            label_col = candidate
            break

    if label_col is None:
        object_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()
        if not object_cols:
            raise ValueError("emotion.csv has no categorical emotion label column")
        label_col = object_cols[0]

    mapped = (
        df[label_col]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(EMOTION_MAP)
        .astype(float)
    )

    out = pd.DataFrame({"emotion_score": scale_1_to_10(mapped)})
    out = out.dropna().reset_index(drop=True)
    return out


def preprocess_stress(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "stress.csv")
    working = df.copy()

    for col in working.columns:
        working[col] = pd.to_numeric(working[col], errors="ignore")

    numeric_cols = working.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("stress.csv has no numeric columns for stress preprocessing")

    survey_keywords = ("survey", "questionnaire", "self", "pss", "score", "stress")
    survey_cols = [
        c
        for c in numeric_cols
        if any(k in c.lower() for k in survey_keywords)
    ]
    physio_cols = [c for c in numeric_cols if c not in survey_cols]

    if not physio_cols:
        physio_cols = [c for c in numeric_cols if c not in survey_cols[:1]]

    norm_cols = sorted(set(physio_cols + survey_cols))
    normalized = normalize_columns(working, norm_cols)

    components = []
    if physio_cols:
        components.append(normalized[physio_cols].mean(axis=1))
    if survey_cols:
        components.append(normalized[survey_cols].mean(axis=1))
    if not components:
        components = [normalized[numeric_cols].mean(axis=1)]

    stress_index = pd.concat(components, axis=1).mean(axis=1)

    out_cols = sorted(set(physio_cols + survey_cols))
    out = normalized[out_cols].copy()
    out["stress_index"] = stress_index
    out = out.dropna().reset_index(drop=True)
    return out


def preprocess_mental_health(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "mental_health.csv")
    working = df.copy()

    for col in working.columns:
        working[col] = pd.to_numeric(working[col], errors="ignore")

    numeric_cols = working.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("mental_health.csv has no numeric trait measures")

    trait_keywords = (
        "anxiety",
        "depression",
        "stress",
        "mood",
        "trait",
        "mental",
        "phq",
        "gad",
        "wellbeing",
    )
    trait_cols = [c for c in numeric_cols if any(k in c.lower() for k in trait_keywords)]
    if not trait_cols:
        trait_cols = numeric_cols

    normalized = normalize_columns(working, trait_cols)
    out = normalized[trait_cols].copy()
    out["trait_load"] = out.mean(axis=1)
    out = out.dropna().reset_index(drop=True)
    return out


def preprocess_eeg(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "eeg.csv")
    working = df.apply(pd.to_numeric, errors="coerce")

    numeric_cols = working.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("eeg.csv has no numeric EEG columns")

    eeg_numeric = working[numeric_cols].copy()

    mean_feature = eeg_numeric.mean(axis=1)
    var_feature = eeg_numeric.var(axis=1)

    pca_feature = None
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        filled = eeg_numeric.fillna(eeg_numeric.mean())
        scaled = StandardScaler().fit_transform(filled)
        pca_feature = pd.Series(PCA(n_components=1).fit_transform(scaled).ravel(), index=working.index)
    except Exception:
        pca_feature = None

    base = pca_feature if pca_feature is not None else (mean_feature + var_feature) / 2.0

    out = pd.DataFrame(
        {
            "eeg_mean": mean_feature,
            "eeg_variance": var_feature,
            "EEG_feature": scale_1_to_10(base),
        }
    )
    out = out.dropna().reset_index(drop=True)
    return out


def combine_cognitive_features(
    emotion_df: pd.DataFrame,
    stress_df: pd.DataFrame,
    mh_df: pd.DataFrame,
    eeg_df: pd.DataFrame,
) -> pd.DataFrame:
    combined = pd.concat(
        [
            emotion_df[["emotion_score"]],
            stress_df[["stress_index"]],
            mh_df[["trait_load"]],
            eeg_df[["EEG_feature"]],
        ],
        axis=1,
        join="inner",
    )

    combined["cognitive_load_final"] = (
        COGNITIVE_WEIGHTS["emotion_score"] * combined["emotion_score"]
        + COGNITIVE_WEIGHTS["stress_index"] * combined["stress_index"]
        + COGNITIVE_WEIGHTS["trait_load"] * combined["trait_load"]
        + COGNITIVE_WEIGHTS["EEG_feature"] * combined["EEG_feature"]
    )

    numeric_cols = combined.select_dtypes(include=[np.number]).columns
    combined[numeric_cols] = combined[numeric_cols].astype(float)
    combined = combined.dropna().reset_index(drop=True)
    return combined


def ensure_float_and_dropna(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna().reset_index(drop=True)
    return out.astype(float)


def save_outputs(data_dir: Path, outputs: dict[str, pd.DataFrame]) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for filename, df in outputs.items():
        df.to_csv(data_dir / filename, index=False)


def run_preprocessing(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    spotify_df = ensure_float_and_dropna(preprocess_spotify(data_dir))
    emotion_df = ensure_float_and_dropna(preprocess_emotion(data_dir))
    stress_df = ensure_float_and_dropna(preprocess_stress(data_dir))
    mh_df = ensure_float_and_dropna(preprocess_mental_health(data_dir))
    eeg_df = ensure_float_and_dropna(preprocess_eeg(data_dir))

    combined_df = combine_cognitive_features(emotion_df, stress_df, mh_df, eeg_df)
    combined_df = ensure_float_and_dropna(combined_df)

    outputs = {
        "spotify_preprocessed.csv": spotify_df,
        "emotion_preprocessed.csv": emotion_df,
        "stress_preprocessed.csv": stress_df,
        "mental_health_preprocessed.csv": mh_df,
        "eeg_preprocessed.csv": eeg_df,
        "cognitive_load_preprocessed.csv": combined_df,
    }
    save_outputs(data_dir, outputs)
    return outputs


if __name__ == "__main__":
    result = run_preprocessing()
    print("Preprocessing complete. Saved files:")
    for name, df in result.items():
        print(f"- {name}: {df.shape[0]} rows x {df.shape[1]} cols")
