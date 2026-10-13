#!/usr/bin/env python3
"""Predict COF band gaps from a CSV containing the xTB4 feature columns."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import joblib
import numpy as np
from tabpfn.model_loading import load_fitted_tabpfn_model


HERE = Path(__file__).resolve().parent


def number(value: str | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_csv", type=Path)
    args = parser.parse_args()

    bundle = joblib.load(HERE / "DFT_replaced_xTB4_bundle.joblib")
    with args.input_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing = [name for name in bundle["feature_names"] if name not in rows[0]]
    if missing:
        raise ValueError(f"Input is missing {len(missing)} required features; first: {missing[:8]}")

    x = np.array(
        [[number(row.get(name)) for name in bundle["feature_names"]] for row in rows]
    )
    x = bundle["imputer"].transform(x)
    tabpfn = load_fitted_tabpfn_model(HERE / bundle["tabpfn_fit_file"], device="cpu")
    tab_pred = tabpfn.predict(x)
    rf_pred = bundle["random_forest"].predict(x)
    weights = bundle["ensemble_weights"]
    ensemble = weights["tabpfn"] * tab_pred + weights["random_forest"] * rf_pred

    output_fields = list(rows[0]) + [
        "predicted_band_gap_eV",
        "tabpfn_prediction_eV",
        "random_forest_prediction_eV",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        for row, final, tab_value, rf_value in zip(rows, ensemble, tab_pred, rf_pred):
            row.update(
                predicted_band_gap_eV=float(final),
                tabpfn_prediction_eV=float(tab_value),
                random_forest_prediction_eV=float(rf_value),
            )
            writer.writerow(row)
    print(f"wrote {len(rows)} predictions to {args.output_csv}")


if __name__ == "__main__":
    main()
