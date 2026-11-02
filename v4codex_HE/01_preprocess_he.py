from pathlib import Path
import shutil

import pandas as pd

import config


def load_he_measurements():
    if not config.HE_DETECTIONS_PATH.exists():
        raise FileNotFoundError(f"HE detection file not found: {config.HE_DETECTIONS_PATH}")

    detections = pd.read_csv(config.HE_DETECTIONS_PATH, sep="\t")
    missing = set(config.HE_COLUMNS).difference(detections.columns)
    if missing:
        raise ValueError(f"Missing HE columns: {sorted(missing)}")

    detections = detections[config.HE_COLUMNS].rename(
        columns={
            "Centroid X µm": "X",
            "Centroid Y µm": "Y",
        }
    )
    return detections.drop_duplicates(subset=["X", "Y"])


def process_region(cell_data_path, he_measurements):
    region_id = cell_data_path.name.split(".")[0]
    region_root = config.REGION_TABLE_ROOT
    required_paths = {
        "cell_data": cell_data_path,
        "cell_types": region_root / f"{region_id}.cell_types.csv",
        "expression": region_root / f"{region_id}.expression.csv",
    }

    missing = [path for path in required_paths.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing files for {region_id}: {missing}")

    cell_data = pd.read_csv(required_paths["cell_data"])
    if not {"CELL_ID", "X", "Y"}.issubset(cell_data.columns):
        raise ValueError(f"Invalid coordinate table for {region_id}")

    matched_features = cell_data[["CELL_ID", "X", "Y"]].merge(
        he_measurements,
        on=["X", "Y"],
        how="inner",
    )
    feature_table = matched_features[["CELL_ID", *config.HE_FEATURE_COLUMNS]]
    feature_table = feature_table.drop_duplicates(subset=["CELL_ID"])
    if feature_table.empty:
        raise ValueError(f"No HE detections matched region {region_id}")

    tables = {
        "cell_data": cell_data,
        "cell_features": feature_table,
        "cell_types": pd.read_csv(required_paths["cell_types"]),
        "expression": pd.read_csv(required_paths["expression"]),
    }

    if any("CELL_ID" not in table.columns for table in tables.values()):
        raise ValueError(f"Every table must contain CELL_ID for {region_id}")

    common_ids = set.intersection(
        *(set(table["CELL_ID"]) for table in tables.values())
    )
    if not common_ids:
        raise ValueError(f"No shared CELL_ID values for {region_id}")

    ordered_ids = cell_data.loc[
        cell_data["CELL_ID"].isin(common_ids), "CELL_ID"
    ].drop_duplicates()
    config.PROCESSED_REGION_ROOT.mkdir(parents=True, exist_ok=True)

    for table_name, table in tables.items():
        filtered = table[table["CELL_ID"].isin(common_ids)].copy()
        filtered["_order"] = filtered["CELL_ID"].map(
            {cell_id: index for index, cell_id in enumerate(ordered_ids)}
        )
        filtered = filtered.sort_values("_order").drop(columns="_order")
        output_name = f"{region_id}.{table_name}.csv"
        filtered.to_csv(config.PROCESSED_REGION_ROOT / output_name, index=False)

    return region_id


def main():
    he_measurements = load_he_measurements()
    cell_data_files = sorted(config.REGION_TABLE_ROOT.glob("*.cell_data.csv"))
    if not cell_data_files:
        raise FileNotFoundError("No region cell-data tables were found")

    processed_regions = []
    for cell_data_path in cell_data_files:
        region_id = process_region(cell_data_path, he_measurements)
        processed_regions.append(region_id)
        print(f"Processed {region_id}")

    labels = pd.read_csv(config.LABEL_PATH)
    labels = labels[labels["REGION_ID"].isin(processed_regions)]
    labels.to_csv(config.PROCESSED_LABEL_PATH, index=False)
    print(f"Saved labels for {len(processed_regions)} regions")


if __name__ == "__main__":
    main()
