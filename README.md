# Spatial cellular graph analysis

This repository contains two related workflows for modeling tissue microenvironments as spatial cellular graphs. Each cell is a node, and edges represent spatial relationships between cells. A graph neural network learns from local cellular neighborhoods.

## PCF

PCF combines cell coordinates, cell types, biomarker expression, and hematoxylin-eosin morphology. The model predicts both the type of a masked center cell and the transfer phenotype of the tissue region.

```text
Cell tables + HE measurements + region labels
                    │
                    ▼
Match cells across modalities
                    │
                    ▼
Build one spatial graph per region
                    │
                    ▼
Sample local cellular neighborhoods
                    │
                    ▼
Train a multi-task graph neural network
                    │
        ┌───────────┴───────────┐
        ▼                       ▼
Masked center-cell type    Region transfer phenotype
        │                       │
        └───────────┬───────────┘
                    ▼
Extract embeddings and define microenvironment states
                    │
                    ▼
Rearrange B-cell and T-cell attributes
                    │
                    ▼
Test the effect on phenotype prediction
                    │
                    ▼
Summarize B-T spatial contacts
```

1. **Align the input tables.** Match hematoxylin-eosin measurements to the existing cells and retain only cells shared by the coordinate, cell-type, expression, and morphology tables.
2. **Build spatial graphs.** Represent each tissue region as a graph. Nodes carry cell identity and available features, while edges carry spatial relationship and distance.
3. **Learn from local neighborhoods.** Sample a neighborhood around each cell. Mask the center-cell type and train the network to recover it from its surroundings. At the same time, predict the region-level transfer phenotype.
4. **Define microenvironment states.** Extract embeddings from the trained model, reduce their dimensions, and cluster recurring local neighborhoods.
5. **Test B-T spatial organization.** Keep cell positions fixed and redistribute B-cell and T-cell attributes. Compare phenotype predictions before and after this change, and summarize B-T contacts across regions.

## Stereo-seq

Stereo-seq uses a second dataset containing cell coordinates and cell types. A model trained on PCF is adapted to this dataset by freezing its graph message-passing layers and fine-tuning the new input and prediction layers.

```text
Stereo-seq cell coordinates and cell types
                    │
                    ▼
Build one spatial graph per region
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
Train a new model      Load the PCF model
directly               │
                       ▼
               Freeze message-passing layers
                       │
                       ▼
               Replace input and prediction layers
                       │
                       ▼
               Fine-tune on metastasis labels
                       │
                       ▼
          Evaluate on held-out regions
```

1. **Build spatial graphs.** Construct one graph per region from cell positions and cell types.
2. **Train a baseline, optionally.** Train a new model directly on Stereo-seq for comparison.
3. **Transfer the PCF model.** Load the pretrained model, preserve its message-passing layers, and replace the layers that depend on the new cell-type vocabulary, feature dimensions, and metastasis labels.
4. **Fine-tune the adaptable layers.** Update only the replaced layers on the Stereo-seq labels while the transferred message-passing layers remain frozen.
5. **Evaluate held-out regions.** Aggregate local predictions within each unseen region and compare them with the region labels.
