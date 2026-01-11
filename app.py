import sys
from pathlib import Path

# --- Project paths ---
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
ARTIFACT_PATH = ROOT / "artifacts" / "HistGradientBoosting_tuned.joblib"
IMPORTANCE_PATH = ROOT / "results" / "feature_importance_permutation_f1.csv"

# Ensure src/ is importable so joblib can unpickle custom transformers
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import altair as alt


st.set_page_config(page_title="Health Risk ML – DRK_YN Demo", layout="centered")


@st.cache_resource
def load_artifact(path: Path) -> dict:
    return joblib.load(path)


@st.cache_data
def load_importance(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    # expected: feature, importance_mean, importance_std
    if "importance_mean" in df.columns:
        df = df.sort_values("importance_mean", ascending=False)
    return df


def build_full_input(feature_columns: list[str], user_values: dict) -> pd.DataFrame:
    """
    Build a 1-row DataFrame with ALL expected raw columns.
    Fill with NaN by default; overwrite only user-provided values.
    This prevents ColumnTransformer from failing due to missing columns.
    """
    row = {col: np.nan for col in feature_columns}
    for k, v in user_values.items():
        if k in row:
            row[k] = v
    return pd.DataFrame([row], columns=feature_columns)


st.title("🍷 Drinking Prediction (DRK_YN) – Streamlit MVP")
st.caption(
    "Portfolio demo: inference using a tuned HistGradientBoosting pipeline "
    "(preprocessing + model) and global permutation feature importance."
)

# --- Load artifact ---
if not ARTIFACT_PATH.exists():
    st.error(f"Model artifact not found at: {ARTIFACT_PATH}")
    st.stop()

artifact = load_artifact(ARTIFACT_PATH)
pipe = artifact.get("model")
model_name = artifact.get("model_name", "Model")
cv_best_f1 = artifact.get("cv_best_f1", None)
feature_columns = artifact.get("feature_columns")

if pipe is None:
    st.error("Artifact does not contain a 'model' key.")
    st.stop()

if not feature_columns:
    st.error(
        "Artifact missing 'feature_columns'.\n\n"
        "Fix: re-save the joblib artifact including `feature_columns = X.columns.tolist()`."
    )
    st.stop()

with st.expander("Model details", expanded=False):
    st.write(f"**Model:** {model_name}")
    if cv_best_f1 is not None:
        st.write(f"**CV best F1 (tuning):** {cv_best_f1:.4f}")
    st.write(f"**Expected raw features:** {len(feature_columns)} columns")


# --- Sidebar inputs (MVP: only a subset) ---
st.sidebar.header("Inputs (subset)")

# NOTE: These names MUST match raw dataset column names.
age = st.sidebar.number_input(
    "age (rounded to 5 years)",
    min_value=0, max_value=120, value=45, step=5
)

height = st.sidebar.number_input(
    "height [cm] (rounded to 5 cm)",
    min_value=0.0, value=170.0, step=5.0
)

weight = st.sidebar.number_input(
    "weight [kg]",
    min_value=0.0, value=70.0, step=1.0
)


sex = st.sidebar.selectbox("sex", ["Male", "Female"], index=0)

gamma_GTP = st.sidebar.number_input(
    "gamma-GTP [IU/L]",
    min_value=0.0, value=30.0, step=1.0
)

SGOT_AST = st.sidebar.number_input(
    "SGOT(AST) [IU/L]",
    min_value=0.0, value=20.0, step=1.0
)

SGOT_ALT = st.sidebar.number_input(
    "SGOT(ALT) [IU/L]",
    min_value=0.0, value=20.0, step=1.0
)

HDL_chole = st.sidebar.number_input(
    "HDL cholesterol [mg/dL]",
    min_value=0.0, value=55.0, step=1.0
)

LDL_chole = st.sidebar.number_input(
    "LDL cholesterol [mg/dL]",
    min_value=0.0, value=120.0, step=1.0
)

triglyceride = st.sidebar.number_input(
    "triglyceride [mg/dL]",
    min_value=0.0, value=110.0, step=1.0
)


waistline = st.sidebar.number_input("waistline", min_value=0.0, value=80.0, step=1.0)

smk_label = st.sidebar.selectbox(
    "smoking status",
    ["Never smoker", "Former smoker", "Current smoker"],
    index=0,
    help="Smoking status as defined in the original dataset.",
)

SMK_stat_type_cd = {"Never smoker": 1, "Former smoker": 2, "Current smoker": 3}[smk_label]


user_values = {
    "age": age,
    "sex": sex,
    "gamma_GTP": gamma_GTP,
    "SGOT_ALT": SGOT_ALT,
    "SGOT_AST": SGOT_AST,
    "HDL_chole": HDL_chole,
    "LDL_chole": LDL_chole,
    "triglyceride": triglyceride,
    "height": height,
    "weight": weight,
    "waistline": waistline,
    "SMK_stat_type_cd": SMK_stat_type_cd,
}

X_input = build_full_input(feature_columns, user_values)

st.subheader("Prediction")
predict_btn = st.button("Predict", type="primary")

if predict_btn:
    try:
        proba = float(pipe.predict_proba(X_input)[:, 1][0])
        pred = int(proba >= 0.5)
        label = "Y (Drinker)" if pred == 1 else "N (Non-drinker)"

        st.metric("Predicted class", label)
        st.metric("Probability of Y", f"{proba:.3f}")

        st.info(
            "This is a portfolio demo. Outputs are model predictions based on the dataset and "
            "are not intended for medical or personal decision-making."
        )
    except Exception as e:
        st.error("Prediction failed due to an unexpected input/pipeline issue.")
        st.exception(e)

st.divider()

# --- Global feature importance panel (from Notebook 05 output) ---
st.subheader("Global feature importance (Permutation)")

imp_df = load_importance(IMPORTANCE_PATH)

if imp_df is None:
    st.warning(f"Importance CSV not found at: {IMPORTANCE_PATH}")
else:
    top_n = 10

    # Top N ordered descending
    top = (
        imp_df.sort_values("importance_mean", ascending=False)
        .head(top_n)
        .copy()
    )

    st.caption(f"Top {top_n} features by mean decrease in F1 when permuted.")

    # Altair chart with explicit sort order
    order = top["feature"].tolist()  # already in descending order

    chart = (
        alt.Chart(top)
        .mark_bar()
        .encode(
            x=alt.X("importance_mean:Q", title="Mean ΔF1 (permutation)"),
            y=alt.Y("feature:N", sort=order, title="Feature"),
            tooltip=[
                alt.Tooltip("feature:N"),
                alt.Tooltip("importance_mean:Q", format=".4f", title="mean"),
                alt.Tooltip("importance_std:Q", format=".4f", title="std"),
            ],
        )
        .properties(height=320)
    )

    st.altair_chart(chart, use_container_width=True)

    with st.expander("Show importance table (mean ± std)", expanded=False):
        top_table = top.copy()
        top_table["mean±std"] = (
            top_table["importance_mean"].round(4).astype(str)
            + " ± "
            + top_table["importance_std"].round(4).astype(str)
        )
        st.dataframe(
            top_table[["feature", "importance_mean", "importance_std", "mean±std"]],
            width="stretch",
            hide_index=True,
        )
