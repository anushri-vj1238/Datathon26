from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).resolve().parent / "data"
SPOTIFY_CANDIDATES = ["spotify_preprocessed.csv", "spotify.csv"]
COGNITIVE_CANDIDATES = [
    "cognitive_load_preprocessed.csv",
    "cognitive_emotional_preprocessed.csv",
    "cognitive_preprocessed.csv",
]

FINAL_COLUMNS = [
    "tempo",
    "energy",
    "valence",
    "danceability",
    "acousticness",
    "loudness",
    "cognitive_load_final",
]


def _load_first_existing(data_dir: Path, candidates: list[str]) -> pd.DataFrame:
    for name in candidates:
        path = data_dir / name
        if path.exists():
            return pd.read_csv(path)
    raise FileNotFoundError(
        f"None of the expected files were found in {data_dir}: {candidates}"
    )


def create_merged_dataset(data_dir: Path = DATA_DIR, random_state: int = 42) -> pd.DataFrame:
    spotify_df = _load_first_existing(data_dir, SPOTIFY_CANDIDATES)
    cognitive_df = _load_first_existing(data_dir, COGNITIVE_CANDIDATES)

    required_spotify = [
        "tempo",
        "energy",
        "valence",
        "danceability",
        "acousticness",
        "loudness",
    ]
    missing_spotify = [c for c in required_spotify if c not in spotify_df.columns]
    if missing_spotify:
        raise ValueError(f"Spotify dataset missing required columns: {missing_spotify}")

    if "cognitive_load_final" not in cognitive_df.columns:
        raise ValueError("Cognitive dataset must include 'cognitive_load_final'")

    spotify_pool = spotify_df[required_spotify].copy()
    spotify_pool = spotify_pool.apply(pd.to_numeric, errors="coerce")

    cognitive = cognitive_df[["cognitive_load_final"]].copy()
    cognitive["cognitive_load_final"] = pd.to_numeric(
        cognitive["cognitive_load_final"], errors="coerce"
    )

    cognitive = cognitive.dropna().reset_index(drop=True)
    spotify_pool = spotify_pool.dropna().reset_index(drop=True)

    if spotify_pool.empty or cognitive.empty:
        raise ValueError("Input datasets must contain at least one non-null row")

    sampled_spotify = spotify_pool.sample(
        n=len(cognitive), replace=True, random_state=random_state
    ).reset_index(drop=True)

    merged = pd.concat([sampled_spotify, cognitive], axis=1)
    merged = merged[FINAL_COLUMNS]

    noise = np.random.default_rng(random_state).normal(loc=0.0, scale=1.0, size=len(merged))
    merged["productivity"] = (
        0.04 * merged["tempo"]
        + 1.2 * merged["energy"]
        - 0.8 * merged["cognitive_load_final"]
        + noise
    )

    merged = merged.dropna().reset_index(drop=True)
    for col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors="coerce").astype(float)
    merged = merged.dropna().reset_index(drop=True)

    data_dir.mkdir(parents=True, exist_ok=True)
    output_path = data_dir / "merged_dataset.csv"
    merged.to_csv(output_path, index=False)
    return merged


if __name__ == "__main__":
    result = create_merged_dataset()
    print(f"Saved {len(result)} rows to {DATA_DIR / 'merged_dataset.csv'}")
