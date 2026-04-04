# Music popularity prediction — my MLOps learning project

## What I made

I built a **regression** pipeline that predicts **track popularity** from tabular features ( Spotify-style metadata). The core flow is simple on purpose so I could focus on the **plumbing** around the model.

Here is what I implemented and why it mattered for my learning:

- **DVC pipeline** (`dvc.yaml`): I split the work into `prepare` and `train` stages with clear inputs and outputs, so I can re-run the same steps and know what depends on what.
- **MLflow** in `train.py`: I log parameters, metrics, and the sklearn model so I can compare runs instead of losing results in terminal output.
- **Optuna + Hydra** (`src/optimize.py`, `config/config.yaml`): I wanted to try **hyperparameter search** with a proper config file instead of hard-coding ranges everywhere.
- **Apache Airflow DAG** (`dags/ml_training_pipeline.py`): I wrapped DVC steps in a DAG, added a **branch** that checks RMSE from `metrics.json`, and if the model is good enough I **register** it in MLflow (staging). That was my way of learning orchestration beyond “run two scripts by hand.”
- **GitHub Actions** (`.github/workflows/main.yaml`): I set up CI that runs **flake8**, **black --check**, tests the DAG, **builds Docker**, runs **`dvc repro`**, and runs pytest including a simple **RMSE quality gate**. I also experimented with **CML** to post metrics and a feature-importance plot on PRs.
- **Docker**: I wrote a **multi-stage** `Dockerfile` that installs dependencies from wheels and defaults to `dvc repro` so the environment is closer to what CI uses.

## What I used

| Area | What I picked |
|------|----------------|
| Language | Python 3.14 (matches my CI / Docker setup) |
| ML | pandas, scikit-learn — **RandomForest** regressor |
| Pipelines | DVC |
| Tracking | MLflow |
| HPO & config | Optuna, Hydra |
| Orchestration | Airflow 2.x |
| Tests / style | pytest, flake8, black |
| CI | GitHub Actions, CML |

## What you need to run my project

- **Python 3.14** (or change the workflow / Dockerfile if you use another version).
- A CSV at **`data/raw/dataset.csv`** with a **`popularity`** column and fields my `prepare.py` expects. I clean rows, drop some id/text columns, one-hot-encode `track_genre`, and split train/test.
- **DVC** on your PATH for local runs (it is also in `requirements.txt`).

## How I run it locally

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt

# After I put data in data/raw/dataset.csv:
dvc repro                  # prepare + train from dvc.yaml
pytest -v                  # quality gate tests need metrics.json from training
```

**Hyperparameter search** (separate from the default DVC graph — I run it when I want to explore Optuna):

```bash
python src/optimize.py
```

**Airflow**: I run it as a learning setup: set `AIRFLOW_HOME`, wire the project root so the DAG’s `cwd` is correct, then load `dags/ml_training_pipeline.py`. The DAG runs `dvc status`, `dvc repro prepare`, `dvc repro train`, then branches on RMSE and may register the latest MLflow run.

## How the repo is organized

```
src/
  prepare.py    # my data prep + split
  train.py      # training, MLflow, metrics + feature importance plot
  optimize.py   # Optuna + Hydra + MLflow
dags/
  ml_training_pipeline.py   # my Airflow DAG
config/
  config.yaml   # Hydra / HPO settings I tuned
tests/
  test_dag.py, test_model.py
```

## What CI does (in my own words)

When I push or open a PR to `main` or `master`, GitHub installs deps, **lints** and **checks formatting**, runs a **DAG integrity** test, **builds the Docker image**, runs **`dvc repro`**, then **pytest** — including a check that RMSE stays under a threshold (`RMSE_THRESHOLD`, I used **16.5** by default). If artifacts exist, CML adds a comment with **metrics** and the **feature importance** image.

## What I learned (short)

- Small, boring models are fine if the goal is to learn **DVC, MLflow, Airflow, and CI**.
- Fixing CI when data or thresholds drift is part of the skill — not a distraction.
- Comparing my “plain” `train.py` path with `optimize.py` helped me see how **config + search** scale the same idea.

## Data

This is for **school / self-study**. If you use a real dataset, **you** are responsible for licensing and sharing rules — I only describe how my code expects the files to look.
