import os
import numpy as np
import pandas as pd
import optuna
import mlflow
import mlflow.sklearn
import hydra
from omegaconf import DictConfig, OmegaConf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib
import argparse
from sklearn.model_selection import KFold
argparse.ArgumentParser._check_help = lambda self, action: None

def load_data(train_path: str, test_path: str, target_col: str):
    """Завантажує підготовлені дані."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    X_train = df_train.drop(target_col, axis=1).values
    y_train = df_train[target_col].values
    X_test = df_test.drop(target_col, axis=1).values
    y_test = df_test[target_col].values
    
    return X_train, y_train, X_test, y_test

def suggest_params(trial: optuna.Trial, cfg: DictConfig):
    """Визначає простір пошуку для гіперпараметрів."""
    space = cfg.hpo.random_forest_regressor
    return {
        "n_estimators": trial.suggest_int("n_estimators", space.n_estimators.low, space.n_estimators.high),
        "max_depth": trial.suggest_int("max_depth", space.max_depth.low, space.max_depth.high),
        "min_samples_split": trial.suggest_int("min_samples_split", space.min_samples_split.low, space.min_samples_split.high),
        "random_state": cfg.seed
    }

def objective_factory(cfg: DictConfig, X_train, y_train, X_test, y_test):
    """Фабрика для створення objective function для Optuna."""
    def objective(trial: optuna.Trial) -> float:
        params = suggest_params(trial, cfg)
        
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number:03d}"):
            mlflow.set_tag("trial_number", trial.number)
            mlflow.set_tag("model_type", cfg.model.type)
            mlflow.set_tag("sampler", cfg.hpo.sampler)
            mlflow.set_tag("developer", "Kordan_Pavlo")
            mlflow.log_params(params)
            
            # Якщо увімкнено крос-валідацію
            if cfg.hpo.use_cv:
                kf = KFold(n_splits=cfg.hpo.cv_folds, shuffle=True, random_state=cfg.seed)
                scores = []
                
                # Проходимося по фолдах
                for train_idx, val_idx in kf.split(X_train):
                    X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
                    y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]
                    
                    model = RandomForestRegressor(**params)
                    model.fit(X_fold_train, y_fold_train)
                    preds = model.predict(X_fold_val)
                    scores.append(np.sqrt(mean_squared_error(y_fold_val, preds)))
                
                # Усереднюємо RMSE по всіх фолдах
                rmse = np.mean(scores)
            
            # Якщо звичайний train/test split (як було раніше)
            else:
                model = RandomForestRegressor(**params)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                rmse = np.sqrt(mean_squared_error(y_test, preds))
            
            mlflow.log_metric("rmse", rmse)
            return rmse
            
    return objective

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    X_train, y_train, X_test, y_test = load_data(
        cfg.data.train_path, 
        cfg.data.test_path, 
        cfg.data.target_column
    )
    
    if cfg.hpo.sampler.lower() == "random":
        sampler = optuna.samplers.RandomSampler(seed=cfg.seed)
    else:
        sampler = optuna.samplers.TPESampler(seed=cfg.seed)

    with mlflow.start_run(run_name="hpo_parent") as parent_run:
        mlflow.set_tag("model_type", cfg.model.type)
        mlflow.log_dict(OmegaConf.to_container(cfg, resolve=True), "config_resolved.json")
        
        study = optuna.create_study(direction=cfg.hpo.direction, sampler=sampler)
        objective = objective_factory(cfg, X_train, y_train, X_test, y_test)
        
        print(f"Починаємо оптимізацію: {cfg.hpo.n_trials} спроб...")
        study.optimize(objective, n_trials=cfg.hpo.n_trials) 
        
        best_trial = study.best_trial
        print(f"Найкраще RMSE: {best_trial.value:.4f}")
        
        mlflow.log_metric("best_rmse", float(best_trial.value))
        mlflow.log_dict(best_trial.params, "best_params.json")
        
        best_model = RandomForestRegressor(**best_trial.params)
        best_model.fit(X_train, y_train)
        
        os.makedirs("models", exist_ok=True)
        joblib.dump(best_model, "models/best_rf_optimized.pkl")
        mlflow.log_artifact("models/best_rf_optimized.pkl") 
        
        if cfg.mlflow.log_model:
            mlflow.sklearn.log_model(best_model, artifact_path="model") 

if __name__ == "__main__":
    main()