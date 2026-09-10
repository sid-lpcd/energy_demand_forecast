import numpy as np
import pandas as pd

from edf.models.distributional import encode_categoricals, ngboost_quantiles, train_ngboost


def _make_dataset(n=300, seed=0):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    day_of_week = pd.Series(rng.integers(0, 7, size=n), name="day_of_week").astype("category")
    X = pd.DataFrame({"x1": x1, "day_of_week": day_of_week})
    y = pd.Series(x1 * 3.0 + rng.normal(scale=0.5, size=n))
    return X, y


def test_encode_categoricals_expands_category_columns_to_numeric():
    X, _ = _make_dataset()
    encoded = encode_categoricals(X)
    assert "x1" in encoded.columns
    assert not any(isinstance(dt, pd.CategoricalDtype) for dt in encoded.dtypes)
    assert any(col.startswith("day_of_week_") for col in encoded.columns)


def test_encode_categoricals_reindexes_onto_reference_columns():
    X_train, _ = _make_dataset(n=300, seed=0)
    encoded_train = encode_categoricals(X_train)

    # A later split missing one category level entirely (e.g. day_of_week == 6).
    X_other = X_train[X_train["day_of_week"] != 6].copy()
    X_other["day_of_week"] = X_other["day_of_week"].cat.remove_unused_categories()
    encoded_other = encode_categoricals(X_other, reference_columns=encoded_train.columns)

    assert list(encoded_other.columns) == list(encoded_train.columns)
    assert (encoded_other["day_of_week_6"] == 0).all()


def test_train_ngboost_fits_and_predicts_reasonable_mean():
    X, y = _make_dataset(n=500, seed=2)
    encoded = encode_categoricals(X)
    model = train_ngboost(encoded, y, n_estimators=100)
    preds = model.predict(encoded.to_numpy(dtype=float))
    # Should track the dominant linear signal (x1 * 3.0) reasonably well.
    assert np.corrcoef(preds, y)[0, 1] > 0.9


def test_ngboost_quantiles_are_monotonically_non_decreasing_in_alpha():
    X, y = _make_dataset(n=500, seed=3)
    encoded = encode_categoricals(X)
    model = train_ngboost(encoded, y, n_estimators=100)

    quantiles = ngboost_quantiles(model, encoded, alphas=[0.1, 0.5, 0.9])
    stacked = pd.concat([quantiles[0.1], quantiles[0.5], quantiles[0.9]], axis=1).to_numpy()
    assert (np.diff(stacked, axis=1) >= 0).all()


def test_ngboost_quantiles_preserve_index():
    X, y = _make_dataset(n=50, seed=4)
    encoded = encode_categoricals(X)
    model = train_ngboost(encoded, y, n_estimators=20)
    quantiles = ngboost_quantiles(model, encoded, alphas=[0.5])
    assert list(quantiles[0.5].index) == list(encoded.index)
