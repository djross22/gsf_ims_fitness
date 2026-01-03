from pathlib import Path
import yaml
import pandas as pd
from joblib import load as joblib_load
from joblib import dump
from sklearn.base import BaseEstimator

def load_df(manifest, key):
    path = manifest.get(key)
    if path is None:
        raise KeyError(f"Manifest missing key: {key}")
    return pd.read_parquet(path)

def load_yaml(manifest, key):
    path = manifest.get(key)
    if path is None:
        raise KeyError(f"Manifest missing key: {key}")
    with open(path) as f:
        return yaml.safe_load(f)

def load_model(manifest, key):
    path = manifest.get(key)
    if path is None:
        return None
    return joblib.load(path)

def sanitize_manifest_for_save(manifest: dict):
    """
    Split runtime manifest into:
      - YAML-safe manifest
      - tables (DataFrames)
      - models (sklearn / joblib-serializable)
    """

    manifest_clean = {}
    tables = {}
    models = {}

    for key, value in manifest.items():

        # --- DataFrames → tables ---
        if isinstance(value, pd.DataFrame):
            tables[key] = value
            manifest_clean[f"{key}_path"] = f"data/{key}.parquet"

        # --- sklearn models → models ---
        elif isinstance(value, BaseEstimator):
            models[key] = value
            manifest_clean[f"{key}_path"] = f"models/{key}.joblib"

        # --- YAML-safe values ---
        else:
            manifest_clean[key] = value

    return manifest_clean, tables, models

def save_state_v1(
    *,
    manifest: dict,
    state_dir,
    tables: dict[str, "pd.DataFrame"] | None = None,
    models: dict[str, object] | None = None,
):
    """
    Save pipeline state in a Nextflow / AWS Batch–friendly layout.

    Parameters
    ----------
    manifest : dict
        YAML-serializable manifest (config, params, execution state)
    state_dir : str | Path
        Root state directory
    tables : dict[str, pd.DataFrame], optional
        Named DataFrames to save as Parquet
    models : dict[str, object], optional
        Named sklearn models to save via joblib
    """

    state_dir = Path(state_dir)
    state_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # 0. SANITIZE (always)
    # ----------------------------
    clean_manifest, auto_tables, auto_models = sanitize_manifest_for_save(manifest)

    # Explicit inputs override auto-discovered ones
    tables = tables or auto_tables
    models = models or auto_models

    # ----------------------------
    # 1. Save manifest.yaml
    # ----------------------------
    manifest_path = state_dir / "manifest.yaml"
    with open(manifest_path, "w") as f:
        yaml.safe_dump(clean_manifest, f, sort_keys=False)

    # ----------------------------
    # 2. Save tables (Parquet)
    # ----------------------------
    if tables:
        data_dir = state_dir / "data"
        data_dir.mkdir(exist_ok=True)

        for name, df in tables.items():
            df.to_parquet(data_dir / f"{name}.parquet")

    # ----------------------------
    # 3. Save models (joblib)
    # ----------------------------
    if models:
        model_dir = state_dir / "models"
        model_dir.mkdir(exist_ok=True)

        for name, model in models.items():
            dump(model, model_dir / f"{name}.joblib")

def load_state_v1(
    state_dir,
    *,
    load_tables: bool = True,
    load_models: bool = True,
):
    """
    Load pipeline state from a state directory.

    Parameters
    ----------
    state_dir : str | Path
        Root state directory
    load_tables : bool
        Whether to load Parquet tables
    load_models : bool
        Whether to load joblib models

    Returns
    -------
    manifest : dict
        Loaded manifest.yaml
    tables : dict[str, pd.DataFrame]
        Loaded Parquet tables (empty if load_tables=False)
    models : dict[str, object]
        Loaded models (empty if load_models=False)
    """
    state_dir = Path(state_dir)

    # ----------------------------
    # 1. Load manifest.yaml
    # ----------------------------
    manifest_path = state_dir / "manifest.yaml"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest.yaml in {state_dir}")

    with open(manifest_path, "r") as f:
        manifest = yaml.safe_load(f)

    # ----------------------------
    # 2. Load tables (Parquet)
    # ----------------------------
    tables = {}
    if load_tables:
        data_dir = state_dir / "data"
        if data_dir.exists():
            for p in data_dir.glob("*.parquet"):
                tables[p.stem] = pd.read_parquet(p)

    # ----------------------------
    # 3. Load models (joblib)
    # ----------------------------
    models = {}
    if load_models:
        model_dir = state_dir / "models"
        if model_dir.exists():
            for p in model_dir.glob("*.joblib"):
                models[p.stem] = joblib_load(p)

    return manifest, tables, models
