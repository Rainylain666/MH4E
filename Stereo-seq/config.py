from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "Stereo-seq"
REGION_TABLE_ROOT = DATA_ROOT / "regions"
LABEL_PATH = DATA_ROOT / "graph_level_tasks.csv"

DATASET_ROOT = PROJECT_ROOT / "outputs" / "Stereo-seq" / "example_dataset"
GRAPH_ROOT = DATASET_ROOT / "graph"
MODEL_ROOT = PROJECT_ROOT / "outputs" / "Stereo-seq" / "models"
ANALYSIS_ROOT = PROJECT_ROOT / "outputs" / "Stereo-seq" / "analysis"
SPLIT_PATH = PROJECT_ROOT / "outputs" / "Stereo-seq" / "region_splits.npz"

CODEX_CHECKPOINT_PATH = PROJECT_ROOT / "checkpoints" / "codex_pretrained.pt"
BASELINE_CHECKPOINT_PATH = MODEL_ROOT / "st_baseline" / "model_final.pt"
FINETUNED_CHECKPOINT_PATH = MODEL_ROOT / "st_finetuned" / "model_final.pt"

SEED = 42
DEVICE = "cuda"
LABEL_COLUMN = "MET_type"
TEST_SIZE = 0.30

SUBGRAPH_SIZE = 3
SUBGRAPH_RADIUS_LIMIT = 200.0
NEIGHBORHOOD_SIZE = 10

NODE_FEATURES = ["cell_type", "neighborhood_composition", "center_coord"]
EDGE_FEATURES = ["edge_type", "distance"]
PREDICTIVE_FEATURES = ["cell_type"]

# Architecture of the CODEX model whose weights are transferred.
CODEX_MODEL = {
    "num_layer": 3,
    "num_node_type": 17,
    "num_feat": 72,
    "emb_dim": 512,
    "num_node_tasks": 16,
    "num_graph_tasks": 4,
    "node_embedding_output": "last",
    "drop_ratio": 0.25,
    "graph_pooling": "max",
    "gnn_type": "gin",
}

CLASS_NAMES = ["PT/RT", "LnMET", "PmMET", "DoMET"]
