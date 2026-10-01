"""Interactive Streamlit UI for the Student Performance & Risk Predictor.

AIML Club - Oriental College of Technology, Bhopal.

Provides a browser-based front end for the starter pipeline in
``train_and_predict.py``. A mentor (or a student) can dial in attendance,
internal assessment marks, assignment submissions, and weekly study hours, then
immediately receive an "On Track" or "Needs Intervention" verdict backed by a
Random Forest feature-importance chart explaining what drove the decision.

The synthetic cohort and model hyper-parameters mirror ``train_and_predict.py``
exactly, so the UI and the command-line pipeline describe the same problem.

Run with:
    streamlit run app.py
"""

import pickle
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Input features, in the exact column order the model was trained with.
FEATURE_COLUMNS: List[str] = [
    "attendance_pct",
    "internal_marks",
    "assignments_submitted",
    "study_hours_week",
]
PRETTY_LABELS: Dict[str, str] = {
    "attendance_pct": "Attendance (%)",
    "internal_marks": "Internal marks (out of 30)",
    "assignments_submitted": "Assignments submitted (of 9)",
    "study_hours_week": "Study hours per week",
}

TARGET_COLUMN = "at_risk"
AT_RISK = 1
ON_TRACK = 0

MODEL_FILENAME = "model.pkl"
MODEL_PATH = Path(__file__).resolve().parent / MODEL_FILENAME

RANDOM_STATE = 42
SAMPLE_SIZE = 600
TEST_SIZE = 0.2
N_ESTIMATORS = 100

# Slider bounds mirror the range of the synthetic training data so that every
# prediction the UI can produce stays in-distribution.
ATTENDANCE_RANGE = (40.0, 100.0)
INTERNAL_MARKS_RANGE = (10.0, 30.0)
ASSIGNMENTS_RANGE = (2, 9)
STUDY_HOURS_RANGE = (2.0, 25.0)


def build_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Generate the synthetic student cohort used to train the risk model.

    The feature ranges and the risk heuristic are copied from
    ``train_and_predict.py`` so both entry points model the same relationship.

    Returns:
        A ``(features, target)`` tuple, where ``features`` holds the four input
        columns and ``target`` is the binary academic-risk flag
        (0 = On Track, 1 = At Risk).
    """
    # A local generator seeded identically to ``np.random.seed(42)`` produces
    # the same number stream without mutating global random state.
    generator = np.random.RandomState(RANDOM_STATE)

    features = pd.DataFrame(
        {
            "attendance_pct": generator.uniform(40, 100, SAMPLE_SIZE),
            "internal_marks": generator.uniform(10, 30, SAMPLE_SIZE),
            "assignments_submitted": generator.randint(2, 10, SAMPLE_SIZE),
            "study_hours_week": generator.uniform(2, 25, SAMPLE_SIZE),
        }
    )

    risk_score = (
        (100 - features["attendance_pct"]) * 0.4
        + (30 - features["internal_marks"]) * 1.5
        - features["study_hours_week"] * 1.2
    )
    target = (risk_score > 35).astype(int)

    return features, target
def train_model() -> RandomForestClassifier:
    """Train the Random Forest risk classifier from scratch.

    Returns:
        A fitted ``RandomForestClassifier``. Its held-out accuracy is attached
        as ``test_accuracy_`` for display in the UI.
    """
    features, target = build_dataset()
    train_features, test_features, train_target, test_target = train_test_split(
        features, target, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    classifier = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
    )
    classifier.fit(train_features, train_target)
    classifier.test_accuracy_ = float(classifier.score(test_features, test_target))

    return classifier


def load_or_train_model() -> RandomForestClassifier:
    """Return a fitted model, training one only when no saved model is present.

    A pickled model dropped next to this file is reused so repeated sessions
    skip training. Nothing is ever written to disk: the UI trains purely
    in-memory unless the maintainer supplies their own ``model.pkl``.

    Returns:
        A fitted ``RandomForestClassifier``.
    """
    if MODEL_PATH.is_file():
        try:
            with MODEL_PATH.open("rb") as model_file:
                return pickle.load(model_file)
        except (OSError, pickle.UnpicklingError, EOFError, AttributeError,
                ImportError, ValueError):
            # A missing, stale, or unreadable pickle should never break the UI.
            pass

    return train_model()


@st.cache_resource(show_spinner="Training the Random Forest model, one moment...")
def get_model() -> RandomForestClassifier:
    """Return the classifier, training at most once per Streamlit session.

    Returns:
        A fitted ``RandomForestClassifier``.
    """
    return load_or_train_model()


def plot_feature_importance(
    model: RandomForestClassifier,
    feature_columns: List[str],
) -> plt.Figure:
    """Render the model's feature importances as a horizontal bar chart.

    Args:
        model: A fitted classifier exposing ``feature_importances_``.
        feature_columns: Column names aligned with the model's inputs.

    Returns:
        A Matplotlib figure ready to hand to ``st.pyplot``.
    """
    importances = np.asarray(model.feature_importances_, dtype=float)
    ranked = np.argsort(importances)
    labels = [feature_columns[index] for index in ranked]

    figure, axes = plt.subplots(figsize=(7.5, 4.0))
    axes.barh(labels, importances[ranked], color="#2E86DE")
    axes.set_xlabel("Relative importance")
    axes.set_title("What drives the risk prediction?")
    axes.spines["top"].set_visible(False)
    axes.spines["right"].set_visible(False)
    figure.tight_layout()

    return figure


def predict_risk(
    model: RandomForestClassifier,
    metrics: Dict[str, float],
) -> Tuple[int, float]:
    """Predict the academic-risk flag for a single student's metrics.

    Args:
        model: A fitted classifier.
        metrics: Mapping of feature name to the value entered in the UI.

    Returns:
        A ``(prediction, risk_probability)`` tuple, where ``prediction`` is
        ``AT_RISK`` or ``ON_TRACK`` and ``risk_probability`` is the model's
        confidence that the student is at risk.
    """
    frame = pd.DataFrame(
        [[metrics[column] for column in FEATURE_COLUMNS]],
        columns=FEATURE_COLUMNS,
    )
    prediction = int(model.predict(frame)[0])
    risk_probability = float(model.predict_proba(frame)[0][AT_RISK])

    return prediction, risk_probability


def render_sidebar() -> Dict[str, float]:
    """Render the input controls and return the entered metrics.

    Slider bounds match the ranges in the synthetic training cohort so that the
    model is never asked to extrapolate.

    Returns:
        A mapping of feature name to the value the user selected.
    """
    st.sidebar.header("Student Metrics")

    attendance_pct = st.sidebar.slider(
        PRETTY_LABELS["attendance_pct"],
        ATTENDANCE_RANGE[0],
        ATTENDANCE_RANGE[1],
        75.0,
        step=0.5,
    )
    internal_marks = st.sidebar.slider(
        PRETTY_LABELS["internal_marks"],
        INTERNAL_MARKS_RANGE[0],
        INTERNAL_MARKS_RANGE[1],
        20.0,
        step=0.5,
    )
    assignments_submitted = st.sidebar.slider(
        PRETTY_LABELS["assignments_submitted"],
        ASSIGNMENTS_RANGE[0],
        ASSIGNMENTS_RANGE[1],
        5,
        step=1,
    )
    study_hours_week = st.sidebar.slider(
        PRETTY_LABELS["study_hours_week"],
        STUDY_HOURS_RANGE[0],
        STUDY_HOURS_RANGE[1],
        8.0,
        step=0.5,
    )

    return {
        "attendance_pct": attendance_pct,
        "internal_marks": internal_marks,
        "assignments_submitted": float(assignments_submitted),
        "study_hours_week": study_hours_week,
    }


def render_verdict(prediction: int, risk_probability: float) -> None:
    """Display the colour-coded risk verdict for the entered metrics.

    Args:
        prediction: ``AT_RISK`` or ``ON_TRACK`` as returned by :func:`predict_risk`.
        risk_probability: Model confidence that the student is at risk.

    Returns:
        None
    """
    if prediction == AT_RISK:
        st.error(
            "### 🔴 Needs Intervention\n"
            "This student is flagged as **at risk**. Schedule a mentoring "
            "session, review attendance and internal assessments, and set a "
            "weekly study plan before the end-semester exam."
        )
    else:
        st.success(
            "### 🟢 On Track\n"
            "This student is **on track**. Keep the current attendance and "
            "study routine going."
        )

    st.metric("Risk probability", f"{risk_probability * 100:.1f}%")
    st.progress(risk_probability)


def main() -> None:
    """Run the Streamlit application.

    Returns:
        None
    """
    st.set_page_config(
        page_title="Student Performance & Risk Predictor",
        page_icon="🎓",
        layout="centered",
    )

    st.title("🎓 Student Performance & Risk Predictor")
    st.caption(
        "AIML Club — Oriental College of Technology, Bhopal · "
        "Random Forest classifier"
    )

    metrics = render_sidebar()
    model = get_model()
    prediction, risk_probability = predict_risk(model, metrics)

    render_verdict(prediction, risk_probability)

    st.subheader("Why the model decided this")
    st.pyplot(plot_feature_importance(model, FEATURE_COLUMNS))
    st.caption(
        "Feature importance shows how much each input contributed to the "
        "model's decisions across the whole training cohort."
    )

    with st.expander("Entered metrics"):
        st.dataframe(
            pd.DataFrame([metrics]).rename(columns=PRETTY_LABELS),
            hide_index=True,
            width="stretch",
        )

    st.info(
        "This demo trains on a synthetic cohort so it runs with no external "
        "data. Swap `build_dataset()` for a CSV loader to score real students."
    )


if __name__ == "__main__":
    main()