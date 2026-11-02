from pathlib import Path

import pandas as pd
import spacegm

import config


def main():
    labels = pd.read_csv(config.PROCESSED_LABEL_PATH)
    region_ids = labels["REGION_ID"].drop_duplicates().tolist()

    config.GRAPH_ROOT.mkdir(parents=True, exist_ok=True)
    config.FIGURE_ROOT.mkdir(parents=True, exist_ok=True)

    for region_id in region_ids:
        cell_data_path = config.PROCESSED_REGION_ROOT / f"{region_id}.cell_data.csv"
        cell_types_path = config.PROCESSED_REGION_ROOT / f"{region_id}.cell_types.csv"
        expression_path = config.PROCESSED_REGION_ROOT / f"{region_id}.expression.csv"
        features_path = config.PROCESSED_REGION_ROOT / f"{region_id}.cell_features.csv"
        graph_path = config.GRAPH_ROOT / f"{region_id}.gpkl"

        required = [
            cell_data_path,
            cell_types_path,
            expression_path,
            features_path,
        ]
        missing = [path for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing files for {region_id}: {missing}")

        if graph_path.exists():
            print(f"Skipping existing graph: {region_id}")
            continue

        spacegm.construct_graph_for_region(
            region_id,
            cell_coords_file=str(cell_data_path),
            cell_types_file=str(cell_types_path),
            cell_biomarker_expression_file=str(expression_path),
            cell_features_file=str(features_path),
            voronoi_file=None,
            graph_output=str(graph_path),
            voronoi_polygon_img_output=None,
            graph_img_output=None,
            figsize=10,
        )
        print(f"Built graph: {region_id}")


if __name__ == "__main__":
    main()
