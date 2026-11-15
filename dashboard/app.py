from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT_DIR = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT_DIR / "models" / "best_model.joblib"
METRICS_PATH = ROOT_DIR / "reports" / "model_metrics.csv"


st.set_page_config(
    page_title="StudentMind AI",
    page_icon="🧠",
    layout="wide",
)

st.title("🧠 StudentMind AI")
st.caption("Prédiction du niveau de risque en santé mentale des étudiants")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_model_metrics():
    return pd.read_csv(METRICS_PATH)


if not MODEL_PATH.is_file():
    st.error(f"Modèle introuvable : {MODEL_PATH}")
    st.stop()

if not METRICS_PATH.is_file():
    st.error(f"Fichier de métriques introuvable : {METRICS_PATH}")
    st.stop()

model = load_model()
metrics = load_model_metrics()

st.sidebar.title("Navigation")
section = st.sidebar.radio("Choisir une section", ["Prédiction", "Performances"])

if section == "Performances":
    st.header("Performances des modèles")
    metrics_display = metrics.copy()
    score_columns = ["Accuracy", "Precision", "Recall", "F1-score"]
    metrics_display[score_columns] = metrics_display[score_columns] * 100
    st.dataframe(
        metrics_display,
        hide_index=True,
        width="stretch",
        column_config={
            column: st.column_config.NumberColumn(
                column,
                format="%.1f%%",
            )
            for column in score_columns
        },
    )
else:
    st.header("Évaluer un profil étudiant")
    st.write(
        "Renseignez les informations ci-dessous pour obtenir une prédiction "
        "du niveau de risque estimé par le modèle."
    )

    with st.form("student_prediction"):
        col1, col2, col3 = st.columns(3)

        with col1:
            age = st.number_input("Âge", min_value=18, max_value=30, value=21, step=1)
            gender = st.selectbox(
                "Genre",
                ["Male", "Female", "Non-binary"],
            )
            year_of_study = st.number_input(
                "Année d'études",
                min_value=1,
                max_value=5,
                value=2,
                step=1,
            )
            sleep_hours = st.number_input(
                "Sommeil moyen par nuit (heures)",
                min_value=3.5,
                max_value=9.5,
                value=7.0,
                step=0.1,
                format="%.1f",
            )
            study_hours = st.number_input(
                "Étude par semaine (heures)",
                min_value=5.0,
                max_value=55.0,
                value=20.0,
                step=0.5,
                format="%.1f",
            )

        with col2:
            gpa = st.number_input(
                "GPA (échelle de 4)",
                min_value=1.5,
                max_value=4.0,
                value=3.0,
                step=0.1,
                format="%.1f",
            )
            social_media_hours = st.number_input(
                "Réseaux sociaux par jour (heures)",
                min_value=0.5,
                max_value=10.0,
                value=3.0,
                step=0.1,
                format="%.1f",
            )
            screen_time_hours = st.number_input(
                "Temps d'écran par jour (heures)",
                min_value=1.0,
                max_value=14.0,
                value=6.0,
                step=0.1,
                format="%.1f",
            )
            physical_activity = st.selectbox(
                "Activité physique",
                ["Low", "Moderate", "High"],
            )
            financial_stress = st.number_input(
                "Stress financier (1–10)",
                min_value=1.0,
                max_value=10.0,
                value=5.0,
                step=0.5,
                format="%.1f",
            )

        with col3:
            family_support = st.number_input(
                "Soutien familial (1–10)",
                min_value=1.0,
                max_value=10.0,
                value=6.0,
                step=0.5,
                format="%.1f",
            )
            attendance_rate = st.number_input(
                "Taux de présence (%)",
                min_value=30.0,
                max_value=100.0,
                value=85.0,
                step=1.0,
                format="%.1f",
            )
            part_time_job = st.selectbox("Emploi à temps partiel", ["No", "Yes"])
            diet_quality = st.selectbox(
                "Qualité de l'alimentation",
                ["Poor", "Average", "Good"],
            )
            mental_health_history = st.selectbox(
                "Antécédent de santé mentale diagnostiqué",
                ["No", "Yes"],
            )

        submitted = st.form_submit_button(
            "Prédire le niveau de risque",
            type="primary",
            width="stretch",
        )

    if submitted:
        student = pd.DataFrame(
            [
                {
                    "Age": age,
                    "Gender": gender,
                    "Year_of_Study": year_of_study,
                    "Sleep_Hours": sleep_hours,
                    "Study_Hours_Per_Week": study_hours,
                    "GPA": gpa,
                    "Social_Media_Hours": social_media_hours,
                    "Screen_Time_Hours": screen_time_hours,
                    "Physical_Activity": physical_activity,
                    "Financial_Stress": financial_stress,
                    "Family_Support": family_support,
                    "Attendance_Rate": attendance_rate,
                    "Part_Time_Job": part_time_job,
                    "Diet_Quality": diet_quality,
                    "Previous_Mental_Health_History": mental_health_history,
                }
            ]
        )
        prediction = model.predict(student)[0]
        probabilities = model.predict_proba(student)[0]

        st.subheader("Résultat")
        st.metric("Niveau de risque prédit", str(prediction))
        st.caption(
            "Cette prédiction est une estimation du modèle, "
            "et ne constitue pas un diagnostic médical."
        )
        st.write("Probabilités estimées par classe")
        probability_table = pd.DataFrame(
            {
                "Niveau de risque": model.classes_,
                "Probabilité": probabilities,
            }
        ).set_index("Niveau de risque")
        st.bar_chart(probability_table)
