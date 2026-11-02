from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
REGION_TABLE_ROOT = DATA_ROOT / "regions"
PROCESSED_REGION_ROOT = DATA_ROOT / "processed_regions"
HE_DETECTIONS_PATH = DATA_ROOT / "he" / "detections.tsv"
LABEL_PATH = DATA_ROOT / "graph_level.tasks.csv"
PROCESSED_LABEL_PATH = DATA_ROOT / "processed_graph_level.tasks.csv"

DATASET_ROOT = PROJECT_ROOT / "outputs" / "example_dataset"
GRAPH_ROOT = DATASET_ROOT / "graph"
FIGURE_ROOT = DATASET_ROOT / "figures"
MODEL_ROOT = PROJECT_ROOT / "outputs" / "models"
ANALYSIS_ROOT = PROJECT_ROOT / "outputs" / "analysis"
SPLIT_PATH = PROJECT_ROOT / "outputs" / "region_splits.npz"
CHECKPOINT_PATH = MODEL_ROOT / "gin_he" / "model_final.pt"

SEED = 42
DEVICE = "cuda"

SUBGRAPH_SIZE = 3
SUBGRAPH_RADIUS_LIMIT = 200.0
NEIGHBORHOOD_SIZE = 10
BIOMARKER_LOWER_BOUND = 0.0
BIOMARKER_UPPER_BOUND = 18.0

EMBEDDING_SAMPLE_COUNT = 32768
PCA_COMPONENTS = 10
MICROENVIRONMENT_COUNT = 8

B_CELL_TYPES = ("B cells", "B cell")
T_CELL_TYPES = ("CD4T", "CD8T", "Tex", "Treg")
PERMUTATION_TARGET_LABEL = 3
PERMUTATION_REGION_ID = "REGION_ID_TO_ANALYZE"

HE_COLUMNS = [
    "Centroid X µm",
    "Centroid Y µm",
    "Nucleus: Area µm^2",
    "Nucleus: Length µm",
    "Nucleus: Circularity",
    "Nucleus: Max diameter µm",
    "Nucleus: Min diameter µm",
    "Nucleus: Solidity",
    "Hematoxylin: Nucleus: Mean",
    "Eosin: Nucleus: Mean",
    "Cell: Area µm^2",
    "Cell: Length µm",
    "Cell: Circularity",
    "Cell: Max diameter µm",
    "Cell: Min diameter µm",
    "Nucleus/Cell area ratio",
    "Hematoxylin: Cell: Mean",
    "Eosin: Cell: Mean",
    "Hematoxylin: Cytoplasm: Mean",
]

HE_FEATURE_COLUMNS = [
    column for column in HE_COLUMNS
    if column not in {"Centroid X µm", "Centroid Y µm"}
]

HE_FEATURE_NAMES = [
    "CELL: AREA ΜM^2",
    "CELL: CIRCULARITY",
    "CELL: LENGTH ΜM",
    "CELL: MAX DIAMETER ΜM",
    "CELL: MIN DIAMETER ΜM",
    "EOSIN: CELL: MEAN",
    "EOSIN: NUCLEUS: MEAN",
    "HEMATOXYLIN: CELL: MEAN",
    "HEMATOXYLIN: CYTOPLASM: MEAN",
    "NUCLEUS: SOLIDITY",
    "HEMATOXYLIN: NUCLEUS: MEAN",
    "NUCLEUS/CELL AREA RATIO",
    "NUCLEUS: AREA ΜM^2",
    "NUCLEUS: CIRCULARITY",
    "NUCLEUS: LENGTH ΜM",
    "NUCLEUS: MAX DIAMETER ΜM",
    "NUCLEUS: MIN DIAMETER ΜM",
]

CENTER_FEATURES = ["cell_type", *HE_FEATURE_NAMES]

NODE_FEATURES = [
    "cell_type",
    *HE_FEATURE_NAMES,
    "biomarker_expression",
    "neighborhood_composition",
    "center_coord",
]

EDGE_FEATURES = ["edge_type", "distance"]
