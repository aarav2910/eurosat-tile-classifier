# EuroSAT Tile Classifier 
### https://eurosat-tile-classifier.streamlit.app/

Classify a large satellite scene tile by tile and explore the result on an interactive map. A Streamlit app loads a trained **ConvNeXtTiny** RGB model, splits a scene into 64×64 tiles, predicts a land-cover class for every tile, and lets you click any tile to see its class (Forest, River, crop, etc.) and confidence.

The repo covers the full workflow: model comparison (Phase 1), a band-selection study (Phase 2), and the final tile-classification app.

## Repository contents

| File | Purpose |
|---|---|
| `app.py` | Streamlit frontend (final phase) |
| `eurosat_convnext_rgb.keras` | Trained RGB ConvNeXtTiny model (stored with **Git LFS**, ~112 MB) |
| `eurosat_model_comparison_phase1.ipynb` | Phase 1: compares four pretrained backbones on EuroSAT RGB |
| `eurosat_band_selection_phase2_final.ipynb` | Phase 2: compares RGB vs RGB+NIR vs all 13 Sentinel-2 bands |
| `requirements.txt` | Python dependencies |
| `.streamlit/config.toml` | Raises the upload size limit to 2 GB |
| `.gitattributes` | Tracks `*.keras` files with Git LFS |

## The app

**Pipeline:** load model → read large satellite image → split into tiles → classify every tile → color-code the tiles → click a tile to inspect it.

- **Color-coded overlay** with adjustable opacity and a legend.
- **Click-to-inspect:** class, confidence, the tile image and a probability bar chart. The Row/Col inputs stay in sync with map clicks.
- **Hover tooltips** on every tile showing class and confidence.
- **Confidence analysis:** High / Medium / Low buckets with adjustable thresholds. Uncertain tiles can be outlined on the map and reviewed from lowest confidence up.
- **Class filter:** highlight only the classes you choose; the rest of the scene is dimmed.
- **CSV export** of all tile predictions.

### Classes

`AnnualCrop`, `Forest`, `HerbaceousVegetation`, `Highway`, `Industrial`, `Pasture`, `PermanentCrop`, `Residential`, `River`, `SeaLake`

### Accepted inputs

| Input | Handling |
|---|---|
| 12/13-band Sentinel-2 GeoTIFF | Only B04/B03/B02 are read and used as true-color RGB |
| GeoTIFF with 3–11 bands | First 3 bands used as RGB |
| Single-band GeoTIFF | Replicated to 3 channels |
| JPG / PNG | Used as RGB directly |

The model is RGB-only: a 13-band file is accepted, but only its red, green and blue bands reach the model. Non-8-bit GeoTIFFs are rescaled to 0–255 using the 1st–99th intensity percentiles.

### Link to Download Free Satellite Images
https://browser.dataspace.copernicus.eu/

## Setup

The model file is stored with Git LFS, so install it **before cloning**. Downloading the repo as a ZIP from GitHub gives you a tiny pointer file instead of the real model.

```bash
git lfs install
git clone https://github.com/aarav2910/eurosat-tile-classifier.git
cd eurosat-tile-classifier
pip install -r requirements.txt
```

If you already cloned without LFS, run `git lfs pull` inside the repo.

## Run

```bash
streamlit run app.py
```

Upload a scene in the sidebar, or paste a **local file path** to skip the upload limit for very large GeoTIFFs. Tiles are classified once and cached, so changing sliders or clicking tiles doesn't re-run the model.

## Model and results

Data: [EuroSAT](https://github.com/phelber/EuroSAT) (Sentinel-2, 10 classes, 64×64 px), downloaded from the Kaggle dataset `apollo2506/eurosat-dataset`.

**Training setup (both phases):** frozen ImageNet-pretrained backbone, then `GlobalAveragePooling2D → Dropout(0.2) → Dense(10, softmax)`; Adam, up to 30 epochs, batch size 32, early stopping on validation loss (patience 5, best weights restored), seed 42.

### Phase 1: backbone comparison (RGB JPEGs)

Split: 18,900 train / 5,400 validation / 2,700 test.

| Model | Test accuracy | Macro F1 | Training time |
|---|---|---|---|
| **ConvNeXtTiny** | **94.33%** | **94.04%** | 3.11 min |
| EfficientNetB0 | 93.15% | 92.92% | 3.12 min |
| ResNet50 | 92.19% | 91.81% | 1.80 min |
| MobileNetV2 | 89.04% | 88.56% | 1.46 min |

ConvNeXtTiny scored highest, so it is the backbone used in the app.

### Phase 2: band selection (ConvNeXtTiny, 13-band GeoTIFFs)

Split: 19,317 train / 5,519 validation / 2,759 test. For 4- and 13-band inputs, a trainable 1×1 convolution ("band adapter") projects the bands down to 3 channels so the pretrained backbone can be reused.

| Bands | Channels | Test accuracy | Macro F1 | Training time |
|---|---|---|---|---|
| Multispectral | 13 | **94.42%** | **94.08%** | 10.48 min |
| RGB + NIR (B08) | 4 | 92.46% | 91.96% | 10.26 min |
| RGB | 3 | 50.60% | 45.75% | 5.91 min |

The RGB row is far below the Phase 1 RGB result (94.33%) for what should be a comparable setup, so treat it as a pipeline artifact rather than evidence that RGB is weak: input scaling in that experiment's model definition likely interacts badly with the frozen backbone's built-in normalization. See the notebook before drawing conclusions from this row.

Results reflect these specific settings (frozen backbone, this dataset split) and are not a universal ranking.

## Notes and limitations

- **Scaling matters.** The model was trained on EuroSAT's 8-bit RGB images, while the app rescales raw GeoTIFF reflectance with a percentile stretch. A different scaling can shift predictions slightly.
- Tiles don't overlap, and pixels that don't fill a complete 64×64 tile are trimmed from the right and bottom edges.
- Each tile gets a single label, so tiles with mixed land cover tend to show lower confidence.
- Very large scenes (thousands of tiles) can be slow on CPU.

## Citation

> Helber, P., Bischke, B., Dengel, A., Borth, D. *EuroSAT: A Novel Dataset and Deep Learning Benchmark for Land Use and Land Cover Classification.* IEEE JSTARS, 2019.
