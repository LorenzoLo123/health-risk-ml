# src/health_risk_ml/preprocessing.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

Number = Union[int, float]


@dataclass
class DomainCleaningTransformer(BaseEstimator, TransformerMixin):
    """
    Deterministic domain cleaning:
    - Replace exact placeholder values with NaN
    - Replace values below conservative lower bounds with NaN
    """
    placeholder_values: Optional[Dict[str, List[Number]]] = None
    lower_bounds: Optional[Dict[str, Number]] = None

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        Xc = X.copy()

        if self.placeholder_values:
            for col, vals in self.placeholder_values.items():
                if col in Xc.columns:
                    Xc.loc[Xc[col].isin(vals), col] = np.nan

        if self.lower_bounds:
            for col, lb in self.lower_bounds.items():
                if col in Xc.columns:
                    s = pd.to_numeric(Xc[col], errors="coerce") if Xc[col].dtype == "object" else Xc[col]
                    Xc.loc[s < lb, col] = np.nan

        return Xc
