import numpy as np
import pandas as pd
import torch
import spacegm
from spacegm.embeddings_analysis import (
    dimensionality_reduction_combo,
    get_composition_heatmap_for_cluster,
    get_embedding,
    get_random_sampled_subgraphs,
)

import config
from common import apply_training_transforms, build_dataset, build_model, load_checkpoint, resolve_device, stable_softmax


def main():
    dataset, labels = build_dataset()
    num_classes = apply_training_transforms(dataset, labels)
    device = resolve_device()
    model = build_model(dataset, num_classes, pooling="mean")
    model = load_checkpoint(model, device=device).to(device)
    model.eval()

    subgraphs = get_random_sampled_subgraphs(
        dataset,
        n_samples=config.EMBEDDING_SAMPLE_COUNT,
        num_workers=4,
    )
    node_embeddings, graph_embeddings, (_, graph_predictions) = get_embedding(
        model,
        subgraphs,
        str(device),
    )

    graph_predictions = np.asarray(graph_predictions).reshape(-1, num_classes)
    graph_probabilities = stable_softmax(graph_predictions)
    predicted_labels = graph_probabilities.argmax(axis=1)
    confidence = graph_probabilities.max(axis=1)

    _, _, cluster_labels, _ = dimensionality_reduction_combo(
        graph_embeddings,
        n_pca_components=config.PCA_COMPONENTS,
        cluster_method="kmeans",
        n_clusters=config.MICROENVIRONMENT_COUNT,
        seed=config.SEED,
    )

    heatmap, enrichment, cluster_predictions, cluster_counts = (
        get_composition_heatmap_for_cluster(
            subgraphs,
            cluster_labels,
            confidence,
        )
    )

    config.ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        config.ANALYSIS_ROOT / "embeddings.npz",
        node_embeddings=node_embeddings,
        graph_embeddings=graph_embeddings,
        graph_probabilities=graph_probabilities,
        cluster_labels=cluster_labels,
        composition_heatmap=heatmap,
        composition_enrichment=enrichment,
        cluster_predictions=cluster_predictions,
        cluster_counts=cluster_counts,
    )

    assignments = pd.DataFrame(
        {
            "region_id": [subgraph.region_id for subgraph in subgraphs],
            "microenvironment": np.asarray(cluster_labels).reshape(-1),
            "predicted_transfer_type": predicted_labels,
            "prediction_confidence": confidence,
        }
    )
    assignments.to_csv(config.ANALYSIS_ROOT / "microenvironment_assignments.csv", index=False)
    print(f"Saved embedding analysis to {config.ANALYSIS_ROOT}")


if __name__ == "__main__":
    main()
