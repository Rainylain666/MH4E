# Spatial Cellular Graph Learning Pipeline

This repository contains a spatial cellular graph workflow for tissue microenvironment analysis. The model uses cell coordinates, cell types, biomarker expression, and HE morphology features to predict region-level transfer phenotypes.

The public pipeline is implemented by the numbered Python scripts in `v4codex_HE/`. The original notebooks are retained as experimental records and are not required for routine execution.

## Pipeline

```text
Region tables + HE measurements + region labels
                         │
                         ▼
01. Synchronize cell IDs and add HE features
                         │
                         ▼
02. Build one spatial graph per region
                         │
                         ▼
03. Train a multi-task GIN model
                         │
                         ├── Masked center-cell type prediction
                         └── Region-level Transfer_type prediction
                         │
                         ▼
04. Extract embeddings and define MicroE states
                         │
                         ▼
05. Test the effect of B-T spatial permutation
                         │
                         ▼
06. Summarize B-T edge density by region
```

## Public pipeline files

| File | Purpose |
|---|---|
| `config.py` | Relative paths, model settings, feature lists, and analysis parameters |
| `common.py` | Shared dataset, model, split, device, and probability utilities |
| `01_preprocess_he.py` | Match HE measurements to cell tables and synchronize `CELL_ID` values |
| `02_build_graphs.py` | Construct and save one graph per region |
| `03_train_gnn.py` | Create splits, train the multi-task GIN, and save the checkpoint |
| `04_analyze_embeddings.py` | Extract graph embeddings and cluster local microenvironments |
| `05_bt_permutation.py` | Compare predictions before and after B-T attribute permutation |
| `06_bt_edge_analysis.py` | Calculate B-T edge counts and densities for every region |

`spacegm` is an external dependency that provides graph construction, dataset handling, GNN training, inference, and embedding utilities.

## Data layout

Use a project-relative data directory. Do not put private patient data in the repository.

```text
data/
├── regions/
│   ├── {REGION_ID}.cell_data.csv
│   ├── {REGION_ID}.cell_types.csv
│   ├── {REGION_ID}.expression.csv
│   └── ...
├── he/
│   └── detections.tsv
└── graph_level.tasks.csv
```

The scripts create the following directories automatically:

```text
outputs/
├── example_dataset/
│   └── graph/
├── models/
├── analysis/
└── region_splits.npz
```

### Required input columns

| File | Required columns |
|---|---|
| `{REGION_ID}.cell_data.csv` | `CELL_ID`, `X`, `Y` |
| `{REGION_ID}.cell_types.csv` | `CELL_ID` and a cell-type column expected by `spacegm` |
| `{REGION_ID}.expression.csv` | `CELL_ID` and expression features |
| `detections.tsv` | QuPath centroid and HE morphology columns listed in `config.py` |
| `graph_level.tasks.csv` | `REGION_ID`, `Transfer_type` |

All tables belonging to the same region must share compatible `CELL_ID` values. Coordinates must use the same units in the cell tables and HE detections.

## Execution

Run the commands from the repository root.

### 1. Prepare HE-enhanced region tables

```powershell
python v4codex_HE/01_preprocess_he.py
```

This script:

1. Loads the QuPath detection table.
2. Selects nuclear, cellular, hematoxylin, and eosin measurements.
3. Matches HE detections to cells using `X` and `Y`.
4. Writes synchronized tables to `data/processed_regions/`.
5. Writes the filtered label table to `data/processed_graph_level.tasks.csv`.

The script does not overwrite the original region tables.

### 2. Build spatial graphs

```powershell
python v4codex_HE/02_build_graphs.py
```

For each labeled region, the script reads the synchronized cell tables and writes:

```text
outputs/example_dataset/graph/{REGION_ID}.gpkl
```

Each graph stores cell nodes, spatial coordinates, cell attributes, and spatial edge attributes.

### 3. Train the multi-task GIN

```powershell
python v4codex_HE/03_train_gnn.py
```

The training script:

1. Loads the graph dataset.
2. Creates stratified train, validation, and test region splits.
3. Masks the center-cell type for the auxiliary node task.
4. Adds the region-level `Transfer_type` label.
5. Applies the v4 HE feature mask.
6. Trains a GIN with three message-passing layers.
7. Saves the split file and model checkpoint.

Default model settings are:

```text
Subgraph radius:       3 hops
Radius limit:          200 micrometers
Embedding dimension:   512
Dropout:               0.25
Batch size:            64
Learning rate:         0.0005
Graph-loss weight:     1.5
Training iterations:   5000
Graph pooling:         mean
```

The checkpoint is saved to:

```text
outputs/models/gin_he/model_final.pt
```

### 4. Analyze embeddings and MicroE states

```powershell
python v4codex_HE/04_analyze_embeddings.py
```

This script samples local subgraphs, extracts graph embeddings, applies PCA and KMeans, and saves:

```text
outputs/analysis/embeddings.npz
outputs/analysis/microenvironment_assignments.csv
```

The default analysis uses 10 PCA components and 8 microenvironment clusters. These values are configured in `config.py`.

### 5. Run B-T permutation analysis

```powershell
python v4codex_HE/05_bt_permutation.py
```

This script performs two downstream perturbation analyses:

- Region-level permutation: redistribute complete B/T node attributes while preserving node coordinates.
- Subgraph-level permutation: exchange B/T attributes within selected local subgraphs while preserving coordinate features.

The target region and target phenotype index are configured in `config.py`:

```python
PERMUTATION_REGION_ID = "REGION_ID_TO_ANALYZE"
PERMUTATION_TARGET_LABEL = 3
```

The summary is written to:

```text
outputs/analysis/bt_permutation_summary.json
```

The phenotype index is zero-based and must match the label encoding in `graph_level.tasks.csv`.

### 6. Calculate B-T edge density

```powershell
python v4codex_HE/06_bt_edge_analysis.py
```

The output is:

```text
outputs/analysis/bt_edge_counts.csv
```

The analysis uses the B-cell and T-cell type lists in `config.py`.

## Feature usage

The HE model schema contains:

- Cell type
- HE morphology and stain-intensity features
- Biomarker expression
- Neighborhood composition
- Center coordinates

The current feature mask uses HE features for the center node and cell types for neighboring nodes. Biomarker expression, neighborhood composition, and coordinates are retained for graph construction and inspection but are not used as predictive node features by the shown training configuration.

The center-cell type is masked before prediction. The model therefore learns from the surrounding cellular context and the permitted center-cell features.

## v3 baseline

The `v3codex` notebooks document the original baseline workflow. It uses prebuilt spatial graphs and primarily relies on cell-type features. The v4 scripts are the recommended public workflow because they include explicit HE preprocessing, graph construction, configurable paths, named splits, and separated analysis stages.

## ST data and frozen transfer learning

`ST_GNN/` is a second workflow for a dataset that contains cell coordinates and cell types but not the HE feature tables. Its graph-level label column is `MET_type`.

The transfer step loads a CODEX-trained checkpoint, replaces the cell-type embedding, feature projection, and prediction heads for the new dataset, and freezes the pretrained GIN message-passing layers. Only the new input layers and task heads are fine-tuned.

Run [ST_GNN/pipeline.ipynb](ST_GNN/pipeline.ipynb) from top to bottom, or run the scripts individually:

```text
ST_GNN/01_build_graphs.py
ST_GNN/02_train_baseline.py      # optional from-scratch baseline
ST_GNN/03_finetune_frozen.py     # frozen CODEX transfer
ST_GNN/04_evaluate.py
```

Place the ST tables in `data/st_gnn/` and the CODEX checkpoint at `checkpoints/codex_pretrained.pt`. The source-model dimensions are declared in `ST_GNN/config.py` and must match that checkpoint. Region splits are aligned by `REGION_ID`, not by the original row position of the label table.

## Reproducibility and release notes

Before publishing a final model or result, verify the following:

- Install a compatible version of `spacegm`.
- Record Python and package versions.
- Confirm the exact cell-type mapping and `Transfer_type` encoding.
- Keep train, validation, and test regions disjoint.
- Save the split file and checkpoint together with the experiment configuration.
- Set the random seed before training and embedding sampling.
- Use the same feature mask and graph pooling method when loading a checkpoint.
- Validate the HE coordinate matching rate for every region.
- Do not commit private raw data, patient identifiers, or unapproved model weights.

The original notebooks contain exploratory cells, alternative hyperparameters, and historical plotting code. They should not be used as an additional execution stage after the numbered scripts.
