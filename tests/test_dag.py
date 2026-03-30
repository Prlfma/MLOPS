import os
import pytest
from airflow.models import DagBag


def test_dag_import():
    dag_folder = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "dags"))

    dag_bag = DagBag(dag_folder=dag_folder, include_examples=False)

    assert (
        len(dag_bag.import_errors) == 0
    ), f"Помилки імпорту DAG: {dag_bag.import_errors}"
