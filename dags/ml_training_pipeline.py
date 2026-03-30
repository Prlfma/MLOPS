import json
import os
from datetime import datetime
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import BranchPythonOperator, PythonOperator
from airflow.operators.empty import EmptyOperator
import mlflow

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def check_model_quality(**kwargs):
    metrics_path = os.path.join(PROJECT_ROOT, 'metrics.json')
    try:
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
        rmse = float(metrics.get('rmse', 999))
        if rmse <= 16.5:
            return 'register_model'
        else:
            return 'stop_pipeline'
    except Exception as e:
        print(f"Помилка читання метрик: {e}")
        return 'stop_pipeline'

def register_best_model(**kwargs):
    mlflow.set_tracking_uri(f"sqlite:///{os.path.join(PROJECT_ROOT, 'mlflow.db')}")
    experiment = mlflow.get_experiment_by_name("CI_Music_Popularity")
    if experiment:
        client = mlflow.tracking.MlflowClient()
        runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"], max_results=1)
        if runs:
            run_id = runs[0].info.run_id
            model_uri = f"runs:/{run_id}/random_forest_model"
            mv = mlflow.register_model(model_uri, "MusicPopularityModel")
            client.transition_model_version_stage(
                name="MusicPopularityModel",
                version=mv.version,
                stage="Staging"
            )
            print(f"Модель версії {mv.version} зареєстровано у Staging.")

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2023, 1, 1),
    'retries': 1,
}

with DAG('ml_training_pipeline', default_args=default_args, schedule_interval=None, catchup=False) as dag:
    check_data = BashOperator(task_id='check_dvc_status', bash_command='dvc status', cwd=PROJECT_ROOT)
    prepare_data = BashOperator(task_id='prepare_data', bash_command='dvc repro prepare', cwd=PROJECT_ROOT)
    train_model = BashOperator(task_id='train_model', bash_command='dvc repro train', cwd=PROJECT_ROOT)
    evaluate_model = BranchPythonOperator(task_id='evaluate_model', python_callable=check_model_quality)
    register_model = PythonOperator(task_id='register_model', python_callable=register_best_model)
    stop_pipeline = EmptyOperator(task_id='stop_pipeline')

    check_data >> prepare_data >> train_model >> evaluate_model
    evaluate_model >> [register_model, stop_pipeline]
