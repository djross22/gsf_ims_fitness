# Pipeline State Directory (V1)

This directory represents the durable execution state of the
Align-TF fitness pipeline. It replaces the legacy pickled Python
object used in earlier versions of the codebase.

Each Nextflow / AWS Batch step reads and updates this directory.

---

## Directory Structure

```
state/
├── manifest.yaml
├── data/
│   ├── *.parquet
├── models/
│   ├── *.joblib
└── metadata/        (optional, future use)
```

```
Data persistance example flow

SAVE:
objects → sanitize → manifest.yaml + data/ + models/

LOAD (option A):
load_state_v1 → manifest + tables + models

LOAD (option B):
manifest.yaml → load_df / load_model as needed
```

---

## File Semantics

### `manifest.yaml`
Contains **only**:
- configuration parameters
- lists and dictionaries
- execution state
- references to data and model files

Must be YAML-serializable.
No DataFrames or model objects are stored here.

---

### `data/*.parquet`
All tabular data produced by the pipeline:
- intermediate tables
- final outputs
- QC summaries

In V1, **all tables are stored as Parquet**.

---

### `models/*.joblib`
All learned models (e.g. scikit-learn regressors).
Models are saved using `joblib` and referenced by path in the manifest.

---

## Design Principles

- No pickle files
- No Python object persistence
- Filesystem = pipeline state
- Manifest acts as an index, not a container
- Compatible with Nextflow caching and AWS Batch execution

---

## Versioning

This layout corresponds to **state format v1**.
Future versions may introduce:
- CSV vs Parquet splits
- metadata validation
- checksums
