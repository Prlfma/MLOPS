import os
import json
import pandas as pd

def test_data_schema_basic():
    data_path = "data/raw/dataset.csv"
    assert os.path.exists(data_path), f"Data not found at {data_path}"
    
    df = pd.read_csv(data_path)
    assert "popularity" in df.columns, "Target column 'popularity' is missing"
    assert df.shape[0] >= 50, "Too few lines for a learning experiment"

def test_artifacts_exist():
    """Post-train тест: перевірка наявності згенерованих файлів."""
    assert os.path.exists("data/models/random_forest_model.pkl"), "Model artifact not found"
    assert os.path.exists("metrics.json"), "metrics.json not found"
    assert os.path.exists("feature_importance.png"), "feature_importance.png not found"

def test_quality_gate():
    """Quality Gate по метриці RMSE."""
    threshold = float(os.getenv("RMSE_THRESHOLD", "16.5"))
    
    with open("metrics.json", "r", encoding="utf-8") as f:
        metrics = json.load(f)
        
    rmse = float(metrics["rmse"])
    assert rmse <= threshold, f"Quality Gate not passed: RMSE {rmse:.4f} > {threshold}"