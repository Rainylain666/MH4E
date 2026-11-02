import numpy as np
import torch

import config
from common import (
    aligned_split,
    apply_transforms,
    build_dataset,
    build_model,
    resolve_device,
    training_kwargs,
)


def main():
    dataset, labels = build_dataset()
    num_classes = apply_transforms(dataset)
    train_indices, test_indices = aligned_split(dataset, labels)
    config.SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.savez(config.SPLIT_PATH, train=train_indices, test=test_indices)

    device = resolve_device()
    model = build_model(
        num_node_type=len(dataset.cell_type_mapping) + 1,
        num_feat=dataset[0].x.shape[1] - 1,
        num_node_tasks=len(dataset.cell_type_mapping),
        num_graph_tasks=num_classes,
        pooling="mean",
    ).to(device)

    model = spacegm_train(model, dataset, device, train_indices, test_indices)
    config.BASELINE_CHECKPOINT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), config.BASELINE_CHECKPOINT_PATH)
    print(f"Saved baseline checkpoint to {config.BASELINE_CHECKPOINT_PATH}")


def spacegm_train(model, dataset, device, train_indices, test_indices):
    import spacegm

    return spacegm.train.train_subgraph(
        model,
        dataset,
        str(device),
        train_inds=train_indices.tolist(),
        valid_inds=test_indices.tolist(),
        **training_kwargs(config.BASELINE_CHECKPOINT_PATH.parent),
    )


if __name__ == "__main__":
    main()
