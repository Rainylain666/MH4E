from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import spacegm
from sklearn.model_selection import train_test_split

import config


def load_labels():
    labels = pd.read_csv(config.LABEL_PATH)
    required = {"REGION_ID", config.LABEL_COLUMN}
    missing = required.difference(labels.columns)
    if missing:
        raise ValueError(f"Missing label columns: {sorted(missing)}")
    if labels["REGION_ID"].duplicated().any():
        raise ValueError("REGION_ID values must be unique")
    return labels


def dataset_kwargs():
    return {
        "transform": [],
        "pre_transform": None,
        "raw_folder_name": "graph",
        "processed_folder_name": "tg_graph",
        "node_features": config.NODE_FEATURES,
        "edge_features": config.EDGE_FEATURES,
        "subgraph_size": config.SUBGRAPH_SIZE,
        "subgraph_source": "on-the-fly",
        "subgraph_allow_distant_edge": True,
        "subgraph_radius_limit": config.SUBGRAPH_RADIUS_LIMIT,
        "neighborhood_size": config.NEIGHBORHOOD_SIZE,
    }


def build_dataset():
    labels = load_labels()
    dataset = spacegm.CellularGraphDataset(str(config.DATASET_ROOT), **dataset_kwargs())
    return dataset, labels


def apply_transforms(dataset):
    labels = load_labels()
    num_classes = labels[config.LABEL_COLUMN].nunique()
    dataset.set_transforms([
        spacegm.AddCenterCellType(dataset),
        spacegm.AddGraphLabel(
            str(config.LABEL_PATH),
            tasks=[config.LABEL_COLUMN],
            num_classes=num_classes,
        ),
        spacegm.FeatureMask(
            dataset,
            use_center_node_features=config.PREDICTIVE_FEATURES,
            use_neighbor_node_features=config.PREDICTIVE_FEATURES,
        ),
    ])
    return num_classes


def build_model(num_node_type, num_feat, num_node_tasks, num_graph_tasks, pooling):
    return spacegm.GNN_pred(
        num_layer=config.SUBGRAPH_SIZE,
        num_node_type=num_node_type,
        num_feat=num_feat,
        emb_dim=512,
        num_node_tasks=num_node_tasks,
        num_graph_tasks=num_graph_tasks,
        node_embedding_output="last",
        drop_ratio=0.25,
        graph_pooling=pooling,
        gnn_type="gin",
    )


def resolve_device():
    if config.DEVICE == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def aligned_split(dataset, labels):
    """Split by region identity after aligning labels to dataset order."""
    label_map = labels.set_index("REGION_ID")[config.LABEL_COLUMN].to_dict()
    region_ids = list(dataset.region_ids)
    missing = [region_id for region_id in region_ids if region_id not in label_map]
    if missing:
        raise ValueError(f"Missing labels for regions: {missing[:5]}")
    targets = np.asarray([label_map[region_id] for region_id in region_ids])
    indices = np.arange(len(region_ids))
    train, test = train_test_split(
        indices,
        test_size=config.TEST_SIZE,
        random_state=config.SEED,
        stratify=targets,
    )
    return np.sort(train), np.sort(test)


def training_kwargs(model_folder):
    model_folder.mkdir(parents=True, exist_ok=True)
    return {
        "batch_size": 64,
        "lr": 0.001,
        "graph_loss_weight": 1.5,
        "num_iterations": 1000,
        "node_task_loss_fn": nn.CrossEntropyLoss(),
        "graph_task_loss_fn": spacegm.models.MulticlassCrossEntropy(),
        "evaluate_fn": [
            spacegm.train.evaluate_by_sampling_subgraphs,
            spacegm.train.evaluate_by_full_graph,
            spacegm.train.save_model_weight,
        ],
        "evaluate_freq": 100,
        "node_task_evaluate_fn": spacegm.inference.cell_type_prediction_evaluate_fn,
        "graph_task_evaluate_fn": spacegm.inference.graph_multiclass_classification_evaluate_fn,
        "full_graph_node_task_evaluate_fn": spacegm.inference.full_graph_cell_type_prediction_evaluate_fn,
        "full_graph_graph_task_evaluate_fn": spacegm.inference.full_graph_graph_multiclass_classification_evaluate_fn,
        "num_eval_iterations": 10,
        "score_file": str(model_folder / "training_scores.txt"),
        "model_folder": str(model_folder),
    }


def transfer_model(source_model, dataset, num_graph_tasks, freeze_gnn=True):
    """Adapt a CODEX model to the ST dataset and freeze its message-passing layers."""
    import copy

    model = copy.deepcopy(source_model)
    embedding_dim = source_model.emb_dim
    num_node_type = len(dataset.cell_type_mapping) + 1
    num_feat = dataset[0].x.shape[1] - 1
    num_node_tasks = len(dataset.cell_type_mapping)
    hidden_dim = model.mult * embedding_dim

    model.gnn.x_embedding = nn.Embedding(num_node_type, embedding_dim)
    model.gnn.feat_embedding = nn.Linear(num_feat, embedding_dim)
    nn.init.xavier_uniform_(model.gnn.x_embedding.weight)
    nn.init.xavier_uniform_(model.gnn.feat_embedding.weight)

    def prediction_head(output_size):
        return nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, output_size),
        )

    model.num_node_tasks = num_node_tasks
    model.num_graph_tasks = num_graph_tasks
    model.node_pred_module = prediction_head(num_node_tasks)
    model.graph_pred_module = prediction_head(num_graph_tasks)

    if freeze_gnn:
        for name, parameter in model.gnn.named_parameters():
            if "x_embedding" not in name and "feat_embedding" not in name:
                parameter.requires_grad = False
    return model


def stable_softmax(logits):
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / probabilities.sum(axis=-1, keepdims=True)


def load_checkpoint(model, checkpoint_path, device):
    checkpoint = Path(checkpoint_path)
    if not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    model.load_state_dict(torch.load(checkpoint, map_location=device))
    return model
