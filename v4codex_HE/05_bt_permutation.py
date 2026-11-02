import copy
import json
import pickle
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import spacegm
from spacegm.embeddings_analysis import get_all_subgraphs, get_embedding, get_random_sampled_subgraphs

import config
from common import apply_training_transforms, build_dataset, build_model, dataset_kwargs, load_checkpoint, resolve_device, stable_softmax


def cell_type_groups(dataset):
    available = set(dataset.cell_type_mapping)
    b_types = [name for name in config.B_CELL_TYPES if name in available]
    t_types = [name for name in config.T_CELL_TYPES if name in available]
    if not b_types or not t_types:
        raise ValueError("Configured B-cell or T-cell types are absent from the dataset")
    return b_types, t_types


def permute_graph_attributes(graph, b_types, t_types, mode="dispersed", seed=42):
    rng = np.random.default_rng(seed)
    modified = copy.deepcopy(graph)
    target_nodes = []
    b_pool = []
    t_pool = []

    for node in graph.nodes:
        cell_type = graph.nodes[node].get("cell_type")
        if cell_type in b_types:
            target_nodes.append(node)
            b_pool.append(copy.deepcopy(graph.nodes[node]))
        elif cell_type in t_types:
            target_nodes.append(node)
            t_pool.append(copy.deepcopy(graph.nodes[node]))

    if not b_pool or not t_pool:
        raise ValueError("The selected graph does not contain both B and T cells")

    if mode == "coherent":
        coordinates = np.asarray([graph.nodes[node]["center_coord"] for node in target_nodes])
        centered = coordinates - coordinates.mean(axis=0)
        _, eigenvectors = np.linalg.eigh(np.cov(centered.T))
        projections = centered @ eigenvectors[:, -1]
        order = np.argsort(projections)
        categories = np.empty(len(target_nodes), dtype=object)
        categories[order[: len(b_pool)]] = "B"
        categories[order[len(b_pool) :]] = "T"
    elif mode == "dispersed":
        categories = np.array(["B"] * len(b_pool) + ["T"] * len(t_pool), dtype=object)
        rng.shuffle(categories)
    else:
        raise ValueError(f"Unknown permutation mode: {mode}")

    rng.shuffle(b_pool)
    rng.shuffle(t_pool)
    b_index = 0
    t_index = 0

    for node, category in zip(target_nodes, categories):
        source_attributes = b_pool[b_index] if category == "B" else t_pool[t_index]
        if category == "B":
            b_index += 1
        else:
            t_index += 1
        coordinate = modified.nodes[node]["center_coord"]
        modified.nodes[node].update(source_attributes)
        modified.nodes[node]["center_coord"] = coordinate

    return modified


def permute_subgraph_attributes(subgraph, b_indices, t_indices, coordinate_indices, seed):
    rng = np.random.default_rng(seed)
    modified = copy.deepcopy(subgraph)
    features = modified.x.detach().cpu().numpy().copy()
    cell_types = features[:, 0].astype(int)
    b_nodes = np.flatnonzero(np.isin(cell_types, b_indices))
    t_nodes = np.flatnonzero(np.isin(cell_types, t_indices))

    if len(b_nodes) == 0 or len(t_nodes) == 0:
        return modified

    target_nodes = np.concatenate([b_nodes, t_nodes])
    categories = np.array(["B"] * len(b_nodes) + ["T"] * len(t_nodes), dtype=object)
    rng.shuffle(categories)
    b_pool = features[b_nodes].copy()
    t_pool = features[t_nodes].copy()
    rng.shuffle(b_pool)
    rng.shuffle(t_pool)

    b_index = 0
    t_index = 0
    for node, category in zip(target_nodes, categories):
        replacement = b_pool[b_index] if category == "B" else t_pool[t_index]
        if category == "B":
            b_index += 1
        else:
            t_index += 1
        original_coordinates = features[node, coordinate_indices].copy()
        features[node] = replacement
        features[node, coordinate_indices] = original_coordinates

    modified.x = torch.as_tensor(features, dtype=modified.x.dtype)
    return modified


def graph_probabilities(model, subgraphs, device, num_classes):
    with torch.no_grad():
        _, _, (_, predictions) = get_embedding(model, subgraphs, str(device), batch_size=64)
    return stable_softmax(np.asarray(predictions).reshape(-1, num_classes))


@contextmanager
def build_modified_dataset(dataset, modified_graph, target_region_id, labels):
    with tempfile.TemporaryDirectory() as temporary_directory:
        temporary_root = Path(temporary_directory)
        temporary_graph_root = temporary_root / "graph"
        temporary_graph_root.mkdir()
        graph_path = temporary_graph_root / f"{target_region_id}.gpkl"
        with graph_path.open("wb") as handle:
            pickle.dump(modified_graph, handle)

        temporary_dataset = spacegm.CellularGraphDataset(
            str(temporary_root),
            cell_type_mapping=dataset.cell_type_mapping,
            cell_type_freq=dataset.cell_type_freq,
            biomarkers=dataset.biomarkers,
            **dataset_kwargs(),
        )
        apply_training_transforms(temporary_dataset, labels)
        yield temporary_dataset


def run_region_permutation(dataset, labels, model, device, num_classes):
    target_region_id = config.PERMUTATION_REGION_ID
    if target_region_id not in dataset.region_ids:
        raise ValueError(f"Unknown permutation region: {target_region_id}")

    region_index = dataset.region_ids.index(target_region_id)
    original_graph = dataset.get_full_nx(region_index)
    b_types, t_types = cell_type_groups(dataset)
    modified_graph = permute_graph_attributes(
        original_graph,
        b_types,
        t_types,
        mode="dispersed",
        seed=config.SEED,
    )

    original_subgraphs = get_all_subgraphs(dataset, inds=[region_index])
    with build_modified_dataset(dataset, modified_graph, target_region_id, labels) as modified_dataset:
        modified_subgraphs = get_all_subgraphs(modified_dataset, inds=[0])
        original_probabilities = graph_probabilities(model, original_subgraphs, device, num_classes)
        modified_probabilities = graph_probabilities(model, modified_subgraphs, device, num_classes)

    target = config.PERMUTATION_TARGET_LABEL
    summary = {
        "region_id": target_region_id,
        "target_label": target,
        "original_mean_probability": float(original_probabilities[:, target].mean()),
        "modified_mean_probability": float(modified_probabilities[:, target].mean()),
    }
    summary["mean_probability_change"] = (
        summary["modified_mean_probability"] - summary["original_mean_probability"]
    )
    return summary


def run_subgraph_permutation(dataset, model, device, num_classes):
    b_types, t_types = cell_type_groups(dataset)
    b_indices = [dataset.cell_type_mapping[name] for name in b_types]
    t_indices = [dataset.cell_type_mapping[name] for name in t_types]
    coordinate_indices = [
        dataset.node_feature_names.index(name)
        for name in ("center_coord-x", "center_coord-y")
        if name in dataset.node_feature_names
    ]

    subgraphs = get_random_sampled_subgraphs(
        dataset,
        n_samples=config.EMBEDDING_SAMPLE_COUNT,
        num_workers=4,
    )
    original_probabilities = graph_probabilities(model, subgraphs, device, num_classes)
    target = config.PERMUTATION_TARGET_LABEL
    selected = np.flatnonzero(original_probabilities.argmax(axis=1) == target)
    selected_subgraphs = [subgraphs[index] for index in selected]
    modified_subgraphs = [
        permute_subgraph_attributes(
            subgraph,
            b_indices,
            t_indices,
            coordinate_indices,
            seed=config.SEED + offset,
        )
        for offset, subgraph in enumerate(selected_subgraphs)
    ]
    if not modified_subgraphs:
        return {"selected_subgraphs": 0}

    modified_probabilities = graph_probabilities(
        model,
        modified_subgraphs,
        device,
        num_classes,
    )
    original_target = original_probabilities[selected, target]
    modified_target = modified_probabilities[:, target]
    return {
        "selected_subgraphs": int(len(selected)),
        "original_mean_probability": float(original_target.mean()),
        "modified_mean_probability": float(modified_target.mean()),
        "mean_probability_change": float((modified_target - original_target).mean()),
    }


def main():
    dataset, labels = build_dataset()
    num_classes = apply_training_transforms(dataset, labels)
    device = resolve_device()
    model = load_checkpoint(build_model(dataset, num_classes, pooling="mean"), device=device).to(device)
    model.eval()

    region_summary = run_region_permutation(dataset, labels, model, device, num_classes)
    subgraph_summary = run_subgraph_permutation(dataset, model, device, num_classes)
    result = {"region_permutation": region_summary, "subgraph_permutation": subgraph_summary}

    config.ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    with open(config.ANALYSIS_ROOT / "bt_permutation_summary.json", "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
