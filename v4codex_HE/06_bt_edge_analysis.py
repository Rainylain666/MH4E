import numpy as np
import pandas as pd

import config
from common import build_dataset


def count_bt_edges(graph, b_types, t_types):
    count = 0
    for left, right in graph.edges:
        left_type = graph.nodes[left].get("cell_type")
        right_type = graph.nodes[right].get("cell_type")
        if (left_type in b_types and right_type in t_types) or (
            left_type in t_types and right_type in b_types
        ):
            count += 1
    return count


def count_bt_cells(graph, b_types, t_types):
    b_count = 0
    t_count = 0
    for node in graph.nodes:
        cell_type = graph.nodes[node].get("cell_type")
        if cell_type in b_types:
            b_count += 1
        elif cell_type in t_types:
            t_count += 1
    return b_count, t_count


def main():
    dataset, _ = build_dataset()
    available_types = set(dataset.cell_type_mapping)
    b_types = [name for name in config.B_CELL_TYPES if name in available_types]
    t_types = [name for name in config.T_CELL_TYPES if name in available_types]

    if not b_types or not t_types:
        raise ValueError("Configured B-cell or T-cell types are absent from the dataset")

    rows = []
    for index, region_id in enumerate(dataset.region_ids):
        graph = dataset.get_full_nx(index)
        edge_count = count_bt_edges(graph, b_types, t_types)
        b_count, t_count = count_bt_cells(graph, b_types, t_types)
        total_cells = b_count + t_count
        rows.append(
            {
                "region_id": region_id,
                "bt_edge_count": edge_count,
                "b_cell_count": b_count,
                "t_cell_count": t_count,
                "bt_cell_count": total_cells,
                "bt_edge_density": edge_count / total_cells if total_cells else np.nan,
            }
        )

    config.ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    output = pd.DataFrame(rows)
    output.to_csv(config.ANALYSIS_ROOT / "bt_edge_counts.csv", index=False)
    print(f"Saved B-T edge statistics to {config.ANALYSIS_ROOT / 'bt_edge_counts.csv'}")


if __name__ == "__main__":
    main()
