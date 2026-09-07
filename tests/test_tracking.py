import mlflow
import pandas as pd

from edf.tracking import experiment_name, log_model_run


def test_experiment_name_is_namespaced_by_horizon():
    assert experiment_name("1d") == "forecast-1d"
    assert experiment_name("7d") == "forecast-7d"


def test_log_model_run_records_params_tags_and_metrics(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    fold_metrics = pd.DataFrame(
        {
            "mae": [10.0, 20.0],
            "rmse": [15.0, 25.0],
            "mape": [1.0, 2.0],
            "mase": [0.5, 0.6],
        },
        index=pd.Index([2023, 2024], name="year"),
    )

    log_model_run(
        horizon_name="1d",
        horizon_periods=48,
        model_name="persistence",
        fold_metrics=fold_metrics,
        tracking_uri=tracking_uri,
    )

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("forecast-1d")
    assert experiment is not None

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    assert len(runs) == 1
    run = runs.iloc[0]

    assert run["tags.horizon"] == "1d"
    assert run["tags.model"] == "persistence"
    assert run["params.horizon_periods"] == "48"
    assert run["params.n_folds"] == "2"
    # The logged metric is the *last* step's value; the mean is kept separately.
    assert run["metrics.mae"] == 20.0
    assert run["metrics.mean_mae"] == 15.0
    assert run["metrics.mean_mase"] == 0.55


def test_log_model_run_merges_extra_tags(tmp_path):
    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    fold_metrics = pd.DataFrame({"mae": [10.0]}, index=pd.Index([2024], name="year"))

    log_model_run(
        horizon_name="7d",
        horizon_periods=336,
        model_name="lightgbm",
        fold_metrics=fold_metrics,
        tracking_uri=tracking_uri,
        extra_tags={"strategy": "recursive"},
    )

    mlflow.set_tracking_uri(tracking_uri)
    experiment = mlflow.get_experiment_by_name("forecast-7d")
    run = mlflow.search_runs(experiment_ids=[experiment.experiment_id]).iloc[0]

    assert run["tags.strategy"] == "recursive"
    assert run["tags.horizon"] == "7d"
