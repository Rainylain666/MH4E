import spacegm
import torch

import config
from common import (
    aligned_split,
    apply_transforms,
    build_dataset,
    build_model,
    load_checkpoint,
    resolve_device,
    training_kwargs,
    transfer_model,
)


def main():
    dataset, labels = build_dataset()
    num_classes = apply_transforms(dataset)
    train_indices, test_indices = aligned_split(dataset, labels)
    device = resolve_device()

    source_model = build_model(
        num_node_type=config.CODEX_MODEL["num_node_type"],
        num_feat=config.CODEX_MODEL["num_feat"],
        num_node_tasks=config.CODEX_MODEL["num_node_tasks"],
        num_graph_tasks=config.CODEX_MODEL["num_graph_tasks"],
        pooling=config.CODEX_MODEL["graph_pooling"],
    )
    source_model = load_checkpoint(source_model, config.CODEX_CHECKPOINT_PATH, device)
    model = transfer_model(source_model, dataset, num_classes, freeze_gnn=True).to(device)

    model = spacegm.train.train_subgraph(
        model,
        dataset,
        str(device),
        train_inds=train_indices.tolist(),
        valid_inds=test_indices.tolist(),
        **training_kwargs(config.FINETUNED_CHECKPOINT_PATH.parent),
    )
    torch.save(model.state_dict(), config.FINETUNED_CHECKPOINT_PATH)
    print(f"Saved fine-tuned checkpoint to {config.FINETUNED_CHECKPOINT_PATH}")


if __name__ == "__main__":
    main()
