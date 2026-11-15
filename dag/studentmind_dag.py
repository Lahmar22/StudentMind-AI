from datetime import datetime

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator


def pipelineComplet():
    from src.entreinement import optimiser_modeles
    from src.preprocessing import pipelineData
    from src.entreinement import trainer_modeles
    
    X_train, X_test, y_train, y_test, preprocessor = pipelineData()
    
    models = trainer_modeles(
        X_train,
        y_train,
        X_test,
        y_test,
        preprocessor
    )

    optimiser_modeles(models, X_train, y_train, X_test, y_test, preprocessor)

with DAG(
    dag_id="studentmind_dag",
    description="StudentMind AI - Machine Learning Pipeline",
    start_date=datetime(2026, 11, 13),
    schedule=None,
    catchup=False,
    tags=["StudentMind", "Machine Learning", "Classification"],
) as dag:
    
    train_model_task = PythonOperator(
        task_id="pipelineComplet",
        python_callable=pipelineComplet
    )
    
    
    train_model_task