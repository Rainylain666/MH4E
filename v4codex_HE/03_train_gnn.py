import random

import numpy as np
import torch
import torch.nn as nn
import spacegm

import config
from common import (
    apply_training_transforms,
    build_dataset,
    build_model,
    create_region_splits,
    finite_feature_check,
    resolve_device,
)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    set_seed(config.SEED)
    dataset, labels = build_dataset()
    finite_feature_check(dataset)
    num_classes = apply_training_transforms(dataset, labels)

    train_indices, valid_indices, test_indices = create_region_splits(
        dataset, labels, seed=config.SEED
    )
    config.SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        config.SPLIT_PATH,
        train=train_indices,
        valid=valid_indices,
        test=test_indices,
    )

    device = resolve_device()
    model = build_model(dataset, num_classes, pooling="mean").to(device)

    config.MODEL_ROOT.mkdir(parents=True, exist_ok=True)
    model_folder = config.MODEL_ROOT / "gin_he"
    model_folder.mkdir(parents=True, exist_ok=True)

    train_kwargs = {
        "batch_size": 64,
        "lr": 0.0005,
        "graph_loss_weight": 1.5,
        "num_iterations": 5000,
        "node_task_loss_fn": nn.CrossEntropyLoss(),
        "graph_task_loss_fn": spacegm.models.MulticlassCrossEntropy(),
        "evaluate_fn": [
            spacegm.train.evaluate_by_sampling_subgraphs,
            spacegm.train.evaluate_by_full_graph,
            spacegm.train.save_model_weight,
        ],
        "evaluate_freq": 500,
        "node_task_evaluate_fn": spacegm.inference.cell_type_prediction_evaluate_fn,
        "graph_task_evaluate_fn": spacegm.inference.graph_multiclass_classification_evaluate_fn,
        "full_graph_node_task_evaluate_fn": spacegm.inference.full_graph_cell_type_prediction_evaluate_fn,
        "full_graph_graph_task_evaluate_fn": spacegm.inference.full_graph_graph_multiclass_classification_evaluate_fn,
        "num_eval_iterations": 10,
        "score_file": str(model_folder / "training_scores.txt"),
        "model_folder": str(model_folder),
    }

    model = spacegm.train.train_subgraph(
        model,
        dataset,
        str(device),
        train_inds=train_indices.tolist(),
        valid_inds=valid_indices.tolist(),
        **train_kwargs,
    )

    torch.save(model.state_dict(), config.CHECKPOINT_PATH)
    print(f"Saved checkpoint to {config.CHECKPOINT_PATH}")
    print(f"Reserved test regions: {len(test_indices)}")


if __name__ == "__main__":
    main()
