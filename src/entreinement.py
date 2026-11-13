from preprocessing import pipelineData
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from scipy.stats import loguniform, randint, uniform
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
import joblib
from sklearn.pipeline import Pipeline
from pathlib import Path

X_train, X_test, y_train, y_test, preprocessor = pipelineData()

def trainer_modeles(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': LogisticRegression(
            C=1.0,
            solver='lbfgs',
            max_iter=1000,
            random_state=42
        ),

        'Random Forest': RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        ),

        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            random_state=42
        )
    }

    results = []


    for name, model in models.items():

        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        class_names = model.classes_

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=class_names
        )

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(y_test, y_pred, average='macro'),
            'Recall': recall_score(y_test, y_pred, average='macro'),
            'F1-Score': f1_score(y_test, y_pred, average='macro'),
            'ROC AUC': roc_auc_score(y_test, model.predict_proba(X_test), multi_class='ovr'),
            'Confusion Matrix': cm
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by='F1-Score',
        ascending=False
    )

    print(results_df)


# trainer_modeles(X_train, y_train, X_test, y_test)

def optimiser_modeles(X_train, y_train, X_test, y_test):
    models = {
        'Logistic Regression': {
            'model': LogisticRegression(
                max_iter=2000,
                random_state=42
            ),
            'params': {
                'C': loguniform(0.001, 100),
                'solver': ['lbfgs', 'newton-cg'],
                'class_weight': [None, 'balanced']
            }
        },

        'Random Forest': {
            'model': RandomForestClassifier(
                random_state=42,
                n_jobs=-1
            ),
            'params': {
                'n_estimators': randint(100, 500),
                'max_depth': [None, 10, 20, 30, 40],
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'max_features': ['sqrt', 'log2'],
                'class_weight': [None, 'balanced']
            }
        },

        'Gradient Boosting': {
            'model': GradientBoostingClassifier(
                random_state=42
            ),
            'params': {
                'n_estimators': randint(100, 500),
                'learning_rate': loguniform(0.01, 0.3),
                'max_depth': randint(2, 8),
                'min_samples_split': randint(2, 20),
                'min_samples_leaf': randint(1, 10),
                'subsample': uniform(0.6, 0.4),
                'max_features': ['sqrt', 'log2', None]
            }
        }
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )
    best_models = {}

    for name, config in models.items():

        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", config["model"])
        ])

        param_distributions = {
            f"model__{parametre}": valeurs
            for parametre, valeurs in config["params"].items()
        }

        random_search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=30,
            cv=cv,
            scoring='f1_macro',
            n_jobs=-1,
            random_state=42,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        best_models[name] = random_search.best_estimator_

        print("Best Parameters:")
        print(random_search.best_params_)

        print("Best CV F1-score:")
        print(random_search.best_score_)

    results = []

    for name, model in best_models.items():

        y_pred = model.predict(X_test)

        results.append({
            'Model': name,
            'Accuracy': accuracy_score(y_test, y_pred),
            'Precision': precision_score(
                y_test, y_pred, average='macro'
            ),
            'Recall': recall_score(
                y_test, y_pred, average='macro'
            ),
            'F1-score': f1_score(
                y_test, y_pred, average='macro'
            )
        })

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by='F1-score',
        ascending=False
    )

    print(results_df)

    best_model_name = results_df.iloc[0]['Model']
    best_model = best_models[best_model_name]

    model_dir = Path(__file__).resolve().parent.parent / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "best_model.joblib"
    joblib.dump(best_model, model_path)
    print(f"Meilleur modèle ({best_model_name}) sauvegardé : {model_path}")

    

optimiser_modeles(X_train, y_train, X_test, y_test)
