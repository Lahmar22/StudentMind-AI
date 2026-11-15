# StudentMind AI

StudentMind AI est une application de diagnostic d'aide à la décision pour la santé mentale des étudiants. Elle exploite un modèle d'apprentissage automatique pour estimer le niveau de risque émotionnel et psychologique d'un profil étudiant à partir de plusieurs indicateurs (âge, sommeil, charge d'études, GPA, stress financier, soutien familial, temps d'écran, etc.).

## Description de l'application

L'application a pour objectif d'aider les établissements universitaires, les conseillers pédagogiques et les services d'accompagnement à identifier les étudiants potentiellement à risque et à prioriser un suivi adapté. Elle ne remplace pas un diagnostic médical professionnel, mais fournit une estimation basée sur des données structurées et un modèle entraîné.

### Fonctionnalités principales
- Prédiction du niveau de risque de santé mentale à partir d'un formulaire utilisateur
- Interface web interactive avec Streamlit
- Affichage des performances des modèles testés (précision, rappel, score F1, etc.)
- Pipeline d'entraînement et d'optimisation de modèles de classification
- Export des métriques de performance dans le dossier `reports/`

## Stack technique
- Python
- Streamlit
- scikit-learn
- pandas
- joblib
- Airflow (pipeline de formation)
- Docker / Docker Compose

## Structure du projet
- `dashboard/app.py` : interface utilisateur Streamlit
- `src/preprocessing.py` : préparation des données et preprocessing
- `src/entreinement.py` : entraînement et optimisation des modèles
- `dag/studentmind_dag.py` : pipeline Airflow pour lancer l'entraînement
- `models/best_model.joblib` : modèle entraîné sauvegardé
- `reports/model_metrics.csv` : métriques de performance
- `data/student_mental_health.csv` : jeu de données source

## Installation

1. Cloner le dépôt
2. Créer un environnement virtuel et installer les dépendances :

```bash
pip install -r requirements.txt
```

## Lancement de l'application

Démarrer l'interface Streamlit :

```bash
streamlit run dashboard/app.py
```

## Lancement avec Docker

```bash
docker-compose up --build
```

## Remarque importante

Cette application fournit une estimation de risque à partir de données comportementales et académiques. Les résultats doivent être interprétés comme un outil d'aide à la décision et non comme un diagnostic médical ou psychologique définitif.
