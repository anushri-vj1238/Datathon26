from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from model_io import load_model

DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = DATA_DIR / "model.joblib"
FEATURE_COLUMNS = ["tempo", "energy", "valence", "cognitive_load_final"]


def recommend_bpm(cognitive_load_final: float) -> str:
    if cognitive_load_final > 7:
        return "Low BPM (60–90)"
    if 4 <= cognitive_load_final <= 7:
        return "Medium BPM (90–120)"
    return "High BPM (120–160)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict productivity score for MindMelody inputs."
    )
    parser.add_argument("tempo", type=float, help="Tempo value")
    parser.add_argument("energy", type=float, help="Energy value")
    parser.add_argument("valence", type=float, help="Valence value")
    parser.add_argument(
        "cognitive_load_final", type=float, help="Cognitive load final value"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    model = load_model(MODEL_PATH)

    input_df = pd.DataFrame(
        [
            {
                "tempo": args.tempo,
                "energy": args.energy,
                "valence": args.valence,
                "cognitive_load_final": args.cognitive_load_final,
            }
        ]
    )
    input_df = input_df[FEATURE_COLUMNS].astype(float)

    prediction = float(model.predict(input_df)[0])
    bpm_category = recommend_bpm(args.cognitive_load_final)

    print(f"Predicted productivity score: {prediction:.4f}")
    print(f"Recommended BPM category: {bpm_category}")


if __name__ == "__main__":
    main()
