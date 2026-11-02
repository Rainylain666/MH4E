import numpy as np
import pandas as pd
import spacegm
from sklearn.metrics import confusion_matrix

import config
from common import (
    aligned_split,
    apply_transforms,
    build_dataset,
    build_model,
    load_checkpoint,
    resolve_device,
    stable_softmax,
)


def region_probabilities(model, dataset, indices, device, num_classes):
    _, graph_predictions = spacegm.inference.collect_predict_for_all_nodes(
        model, dataset, str(device), inds=indices, print_progress=False
    )
    rows = []
    for index in indices:
        logits = np.asarray(graph_predictions[index]).reshape(-1, num_classes)
        probability = stable_softmax(logits).mean(axis=0)
        rows.append((dataset[index].region_id, probability))
    return rows


def main():
    dataset, labels = build_dataset()
    num_classes = apply_transforms(dataset)
    _, test_indices = aligned_split(dataset, labels)
    device = resolve_device()

    model = build_model(
        num_node_type=len(dataset.cell_type_mapping) + 1,
        num_feat=dataset[0].x.shape[1] - 1,
        num_node_tasks=len(dataset.cell_type_mapping),
        num_graph_tasks=num_classes,
        pooling="max",
    )
    model = load_checkpoint(model, config.FINETUNED_CHECKPOINT_PATH, device).to(device)
    model.eval()

    label_map = labels.set_index("REGION_ID")[config.LABEL_COLUMN].to_dict()
    predictions = region_probabilities(model, dataset, test_indices, device, num_classes)
    true_labels = [label_map[region_id] for region_id, _ in predictions]
    predicted_labels = [int(np.argmax(probability)) for _, probability in predictions]

    names = config.CLASS_NAMES if len(config.CLASS_NAMES) == num_classes else list(range(num_classes))
    matrix = confusion_matrix(true_labels, predicted_labels, labels=list(range(num_classes)))
    config.ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(matrix, index=names, columns=names).to_csv(
        config.ANALYSIS_ROOT / "confusion_matrix.csv"
    )
    print(f"Saved confusion matrix to {config.ANALYSIS_ROOT / 'confusion_matrix.csv'}")


if __name__ == "__main__":
    main()
