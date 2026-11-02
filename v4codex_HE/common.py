from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import spacegm
from sklearn.model_selection import train_test_split

import config


def load_labels(path=None):
    label_path = Path(path or config.PROCESSED_LABEL_PATH)
    if not label_path.exists():
        label_path = config.LABEL_PATH
    labels = pd.read_csv(label_path)
    labels.attrs["source_path"] = str(label_path)
    required = {"REGION_ID", "Transfer_type"}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    return labels


def dataset_kwargs(subgraph_source="on-the-fly"):
    return {
        "transform": [],
        "pre_transform": None,
        "raw_folder_name": "graph",
        "processed_folder_name": "tg_graph",
        "node_features": config.NODE_FEATURES,
        "edge_features": config.EDGE_FEATURES,
        "subgraph_size": config.SUBGRAPH_SIZE,
        "subgraph_source": subgraph_source,
        "subgraph_allow_distant_edge": True,
        "subgraph_radius_limit": config.SUBGRAPH_RADIUS_LIMIT,
        "biomarker_expression_process_method": "linear",
        "biomarker_expression_lower_bound": config.BIOMARKER_LOWER_BOUND,
        "biomarker_expression_upper_bound": config.BIOMARKER_UPPER_BOUND,
        "neighborhood_size": config.NEIGHBORHOOD_SIZE,
    }


def build_dataset(subgraph_source="on-the-fly"):
    labels = load_labels()
    dataset = spacegm.CellularGraphDataset(
        str(config.DATASET_ROOT),
        **dataset_kwargs(subgraph_source=subgraph_source),
    )
    return dataset, labels


def apply_training_transforms(dataset, labels):
    num_classes = labels["Transfer_type"].nunique()
    transforms = [
        spacegm.AddCenterCellType(dataset),
        spacegm.AddGraphLabel(
            str(labels.attrs.get("source_path", config.PROCESSED_LABEL_PATH)),
            tasks=["Transfer_type"],
            num_classes=num_classes,
        ),
        spacegm.FeatureMask(
            dataset,
            use_center_node_features=config.CENTER_FEATURES,
            use_neighbor_node_features=["cell_type"],
        ),
    ]
    dataset.set_transforms(transforms)
    return num_classes


def build_model(dataset, num_graph_tasks, pooling="mean"):
    model = spacegm.GNN_pred(
        num_layer=dataset.subgraph_size,
        num_node_type=len(dataset.cell_type_mapping) + 1,
        num_feat=dataset[0].x.shape[1] - 1,
        emb_dim=512,
        num_node_tasks=len(dataset.cell_type_mapping),
        num_graph_tasks=num_graph_tasks,
        node_embedding_output="last",
        drop_ratio=0.25,
        graph_pooling=pooling,
        gnn_type="gin",
    )
    return model


def resolve_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def region_indices_and_labels(dataset, labels):
    label_map = labels.set_index("REGION_ID")["Transfer_type"].to_dict()
    region_ids = [dataset[i].region_id for i in range(len(dataset))]
    missing = [region_id for region_id in region_ids if region_id not in label_map]
    if missing:
        raise ValueError(f"Missing labels for regions: {missing[:5]}")
    values = np.asarray([label_map[region_id] for region_id in region_ids])
    indices = np.arange(len(region_ids))
    return indices, values


def create_region_splits(dataset, labels, seed=config.SEED):
    indices, targets = region_indices_and_labels(dataset, labels)
    train_valid, test = train_test_split(
        indices,
        test_size=0.20,
        random_state=seed,
        stratify=targets,
    )
    train_targets = targets[train_valid]
    train, valid = train_test_split(
        train_valid,
        test_size=0.25,
        random_state=seed,
        stratify=train_targets,
    )
    return train, valid, test


def stable_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def load_checkpoint(model, checkpoint_path=None, device=None):
    checkpoint = Path(checkpoint_path or config.CHECKPOINT_PATH)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device or "cpu"))
    return model


def finite_feature_check(dataset):
    dataset.set_transforms([])
    for index in range(len(dataset)):
        features = dataset.get_full(index).x
        if not torch.isfinite(features).all():
            raise ValueError(f"Non-finite node features found in dataset index {index}")
