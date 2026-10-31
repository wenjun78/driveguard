"""The model artefact: an XGBoost booster plus the feature specification it was trained with.

Packaged as an MLflow *pyfunc* so the same object serves everywhere - batch scoring through Spark, a Databricks
Model Serving endpoint, or a local call - and so feature preparation (column order, categorical vocabulary,
unseen-model handling) travels with the weights instead of living in a notebook.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field

import mlflow.pyfunc
import numpy as np
import pandas as pd
import xgboost as xgb

CATEGORICAL = "model"


@dataclass
class FeatureSpec:
    numeric: list[str]
    model_vocab: dict[str, int]  # drive model -> code; unseen -> -1
    model_freq: dict[str, float]  # drive model -> share of the training fleet
    horizon_days: int
    alert_budget: float
    extras: dict = field(default_factory=dict)

    @property
    def columns(self) -> list[str]:
        return self.numeric + ["model_code", "model_freq"]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=1)

    @classmethod
    def from_json(cls, text: str) -> FeatureSpec:
        return cls(**json.loads(text))


def prepare(pdf: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Build the model matrix from a gold-shaped frame. Missing numeric columns become NaN (XGBoost handles them)."""
    x = pd.DataFrame(index=pdf.index)
    for c in spec.numeric:
        x[c] = pd.to_numeric(pdf[c], errors="coerce").astype("float32") if c in pdf.columns else np.float32(np.nan)
    m = pdf[CATEGORICAL].astype(str) if CATEGORICAL in pdf.columns else pd.Series("UNKNOWN", index=pdf.index)
    x["model_code"] = m.map(spec.model_vocab).fillna(-1).astype("float32")
    x["model_freq"] = m.map(spec.model_freq).fillna(0.0).astype("float32")
    return x[spec.columns]


class DriveGuardModel:
    def __init__(self, booster: xgb.Booster, spec: FeatureSpec):
        self.booster = booster
        self.spec = spec

    # ---- inference ------------------------------------------------------------------------------------------------
    def predict_proba(self, pdf: pd.DataFrame) -> np.ndarray:
        return self.booster.inplace_predict(prepare(pdf, self.spec).to_numpy(), validate_features=False)

    def reason_codes(self, pdf: pd.DataFrame, top: int = 3) -> list[list[str]]:
        """Top contributing features per row (SHAP-style TreeSHAP contributions from XGBoost itself)."""
        x = prepare(pdf, self.spec)
        contrib = self.booster.predict(xgb.DMatrix(x, feature_names=self.spec.columns), pred_contribs=True)[:, :-1]
        order = np.argsort(-contrib, axis=1)[:, :top]
        cols = np.array(self.spec.columns)
        out = []
        for i, idx in enumerate(order):
            out.append([f"{cols[j]}={x.iloc[i, j]:.4g}" for j in idx if contrib[i, j] > 0])
        return out

    # ---- persistence ----------------------------------------------------------------------------------------------
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        self.booster.save_model(os.path.join(directory, "booster.ubj"))
        with open(os.path.join(directory, "feature_spec.json"), "w") as fh:
            fh.write(self.spec.to_json())

    @classmethod
    def load(cls, directory: str) -> DriveGuardModel:
        booster = xgb.Booster()
        booster.load_model(os.path.join(directory, "booster.ubj"))
        with open(os.path.join(directory, "feature_spec.json")) as fh:
            spec = FeatureSpec.from_json(fh.read())
        return cls(booster, spec)

    def to_bytes(self) -> tuple[bytes, str]:
        return bytes(self.booster.save_raw("ubj")), self.spec.to_json()

    @classmethod
    def from_bytes(cls, raw: bytes, spec_json: str) -> DriveGuardModel:
        booster = xgb.Booster()
        booster.load_model(bytearray(raw))
        return cls(booster, FeatureSpec.from_json(spec_json))


class PyFuncWrapper(mlflow.pyfunc.PythonModel):
    """MLflow pyfunc: input is a gold-shaped DataFrame, output is the failure probability within the horizon."""

    def load_context(self, context):
        self.model = DriveGuardModel.load(context.artifacts["model_dir"])

    def predict(self, context, model_input: pd.DataFrame, params=None):
        return pd.Series(self.model.predict_proba(model_input), name="p_fail_within_horizon")
