from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def save_model(model: Any, path: str | Path) -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import joblib  # type: ignore

        joblib.dump(model, file_path)
        return
    except ModuleNotFoundError:
        pass

    with file_path.open("wb") as f:
        pickle.dump(model, f)


def load_model(path: str | Path) -> Any:
    file_path = Path(path)

    try:
        import joblib  # type: ignore

        return joblib.load(file_path)
    except ModuleNotFoundError:
        pass

    with file_path.open("rb") as f:
        return pickle.load(f)
