"""
EuroSAT Tile Classifier -- Streamlit frontend (RGB model).

Loads a trained RGB ConvNeXtTiny model, tiles a large RGB satellite scene,
classifies every tile, shows a color-coded overlay, lets you click a tile
to see its predicted class + confidence, flags low-confidence tiles for
review, and lets you filter the map down to specific classes.

Run with:
    streamlit run app.py

Place this next to app.py (not uploaded through the UI):
    eurosat_convnext_rgb.keras   -- the trained model (required)

Requires (see requirements.txt):
    streamlit, tensorflow, rasterio, pillow, numpy, pandas, plotly
"""

import os
import io
import json

import numpy as np
import pandas as pd
import rasterio
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import tensorflow as tf
from PIL import Image

st.set_page_config(page_title="EuroSAT Tile Classifier (RGB)", layout="wide")

# -----------------------------------------------------------------------
# Config -- MUST match how the RGB model was trained.
# -----------------------------------------------------------------------
MODEL_PATH = "eurosat_convnext_rgb.keras"
LABEL_MAP_PATH = "label_map.json"

TILE_SIZE = 64

DEFAULT_LABELS = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
                   "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]

CLASS_COLORS = {
    "AnnualCrop":           (255, 221, 113),
    "Forest":               (31, 119, 61),
    "HerbaceousVegetation": (140, 198, 101),
    "Highway":              (120, 120, 120),
    "Industrial":           (197, 60, 60),
    "Pasture":              (188, 222, 150),
    "PermanentCrop":        (214, 158, 46),
    "Residential":          (220, 90, 160),
    "River":                (60, 140, 230),
    "SeaLake":              (20, 70, 160),
}


# -----------------------------------------------------------------------
# Cached loaders -- the model/image/classification only re-run when the
# underlying file/bytes change, not on every click-triggered rerun.
# -----------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading ConvNeXtTiny model...")
def load_classifier(model_path):
    return tf.keras.models.load_model(model_path)


@st.cache_data(show_spinner=False)
def load_label_map(label_map_path):
    with open(label_map_path) as f:
        label_map = json.load(f)
    if all(str(k).isdigit() for k in label_map.keys()):
        idx_to_name = {int(k): v for k, v in label_map.items()}
    else:
        idx_to_name = {v: k for k, v in label_map.items()}
    return [idx_to_name[i] for i in sorted(idx_to_name)]


@st.cache_data(show_spinner="Reading satellite scene...")
def read_scene(file_bytes, filename):
    """Returns an (H, W, 3) uint8 RGB array, whatever the input format."""
    ext = filename.lower().rsplit(".", 1)[-1]
    if ext in ("tif", "tiff"):
        with rasterio.io.MemoryFile(file_bytes) as memfile:
            with memfile.open() as src:
                arr = src.read()  # (bands, H, W)
        if arr.shape[0] >= 3:
            arr = arr[:3]
        elif arr.shape[0] == 1:
            arr = np.repeat(arr, 3, axis=0)
        else:
            raise ValueError(f"Unexpected band count in GeoTIFF: {arr.shape[0]}")
        img = np.transpose(arr, (1, 2, 0))
        if img.dtype != np.uint8:
            img = img.astype("float32")
            lo, hi = np.percentile(img, [1, 99])
            if hi <= lo:
                hi = lo + 1
            img = np.clip((img - lo) / (hi - lo), 0, 1) * 255.0
        img = img.astype("uint8")
    else:
        img = np.array(Image.open(io.BytesIO(file_bytes)).convert("RGB"), dtype="uint8")
    return img


@st.cache_data(show_spinner="Classifying every tile...")
def classify_scene(_model, image_array, tile_size, labels_tuple):
    """image_array: (H, W, 3) uint8 RGB, already cropped to a tile_size multiple."""
    labels = list(labels_tuple)
    h, w, _ = image_array.shape
    n_rows, n_cols = h // tile_size, w // tile_size

    tiles = []
    for r in range(n_rows):
        for c in range(n_cols):
            tile = image_array[r * tile_size:(r + 1) * tile_size,
                                c * tile_size:(c + 1) * tile_size, :]
            tiles.append(tile)
    batch = np.stack(tiles, axis=0)

    probs = _model.predict(batch, verbose=0)
    pred_idx = np.argmax(probs, axis=1)

    results = []
    k = 0
    for r in range(n_rows):
        for c in range(n_cols):
            results.append({
                "row": r,
                "col": c,
                "pred_idx": int(pred_idx[k]),
                "pred_label": labels[pred_idx[k]],
                "confidence": float(probs[k, pred_idx[k]]),
                "probs": probs[k].tolist(),
            })
            k += 1
    return results, n_rows, n_cols


# -----------------------------------------------------------------------
# Confidence buckets
# -----------------------------------------------------------------------
def bucket_confidence(confidence, high_thresh, med_thresh):
    if confidence >= high_thresh:
        return "High"
    elif confidence >= med_thresh:
        return "Medium"
    return "Low"


# -----------------------------------------------------------------------
# Image/overlay building
# -----------------------------------------------------------------------
def build_overlay_image(rgb_display, results, tile_size, alpha, visible_classes):
    """Tiles whose predicted class is in visible_classes get the usual color
    tint. Everything else is desaturated + dimmed so the selected classes
    visually pop out of the scene."""
    overlay = rgb_display.astype("float32").copy()
    for res in results:
        r, c = res["row"], res["col"]
        y0, y1 = r * tile_size, (r + 1) * tile_size
        x0, x1 = c * tile_size, (c + 1) * tile_size

        if res["pred_label"] in visible_classes:
            color = np.array(CLASS_COLORS.get(res["pred_label"], (255, 255, 255)), dtype="float32")
            overlay[y0:y1, x0:x1] = (1 - alpha) * overlay[y0:y1, x0:x1] + alpha * color
        else:
            tile = overlay[y0:y1, x0:x1]
            gray = tile.mean(axis=-1, keepdims=True)
            overlay[y0:y1, x0:x1] = np.repeat(gray, 3, axis=-1) * 0.55

    return overlay.astype("uint8")


def build_tile_hover_trace(results, n_rows, n_cols, tile_size):
    """Invisible Heatmap, one cell per tile, sized exactly to each tile's
    boundaries in *data units* -- it covers the whole tile for hover,
    unlike marker-based traces whose size is set in screen pixels and can't
    be made to exactly match a tile's on-screen footprint."""
    z = np.zeros((n_rows, n_cols))
    customdata = np.empty((n_rows, n_cols, 2), dtype=object)
    for res in results:
        r, c = res["row"], res["col"]
        z[r, c] = res["pred_idx"]
        customdata[r, c, 0] = res["pred_label"]
        customdata[r, c, 1] = f"{res['confidence'] * 100:.1f}%"

    x_centers = [c * tile_size + tile_size / 2 for c in range(n_cols)]
    y_centers = [r * tile_size + tile_size / 2 for r in range(n_rows)]

    return go.Heatmap(
        x=x_centers, y=y_centers, z=z,
        customdata=customdata,
        hovertemplate="Predicted: %{customdata[0]}<br>Confidence: %{customdata[1]}<extra></extra>",
        opacity=0.0,
        showscale=False,
        hoverongaps=True,
    )


def build_tile_click_trace(results, n_rows, n_cols, tile_size):
    """Invisible Scatter marker per tile -- handles click only. Heatmap
    doesn't fire a single-click "selection" event in Plotly (only box/lasso
    -select does), which is why clicking didn't update anything before this
    trace was added. It also provides the tile tooltip so the clickable points
    don't block hover information from another trace."""
    xs, ys, tile_data = [], [], []
    for res in results:
        r, c = res["row"], res["col"]
        xs.append(c * tile_size + tile_size / 2)
        ys.append(r * tile_size + tile_size / 2)
        tile_data.append((r, c, res["pred_label"], f"{res['confidence'] * 100:.1f}%"))

    return go.Scatter(
        x=xs, y=ys,
        customdata=tile_data,
        mode="markers",
        marker=dict(size=tile_size * 0.9, symbol="square",
                    color="rgba(0,0,0,0.001)", line=dict(width=0)),
        hovertemplate="Predicted: %{customdata[2]}<br>Confidence: %{customdata[3]}<extra></extra>",
        showlegend=False,
    )


def sync_selected_tile():
    """Copy the chart's latest point selection into the inspector widgets."""
    chart_state = st.session_state.get("scene_map", {})
    selection = chart_state.get("selection", {})
    points = selection.get("points", [])
    if points:
        row, col = points[0]["customdata"][:2]
        st.session_state["selected_row"] = int(row)
        st.session_state["selected_col"] = int(col)


def make_grid_shapes(n_rows, n_cols, tile_size):
    shapes = []
    for r in range(n_rows + 1):
        y = r * tile_size
        shapes.append(dict(type="line", x0=0, x1=n_cols * tile_size, y0=y, y1=y,
                            line=dict(color="rgba(255,255,255,0.35)", width=1)))
    for c in range(n_cols + 1):
        x = c * tile_size
        shapes.append(dict(type="line", x0=x, x1=x, y0=0, y1=n_rows * tile_size,
                            line=dict(color="rgba(255,255,255,0.35)", width=1)))
    return shapes


def make_highlight_shape(row, col, tile_size):
    """Bright rectangle marking the currently selected tile on the map."""
    y0, y1 = row * tile_size, (row + 1) * tile_size
    x0, x1 = col * tile_size, (col + 1) * tile_size
    return dict(
        type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
        line=dict(color="#FFEE00", width=3),
        fillcolor="rgba(0,0,0,0)",
    )


def make_uncertainty_shape(row, col, tile_size, color):
    """Dashed border marking a low/medium-confidence tile for review."""
    y0, y1 = row * tile_size, (row + 1) * tile_size
    x0, x1 = col * tile_size, (col + 1) * tile_size
    return dict(
        type="rect", x0=x0, x1=x1, y0=y0, y1=y1,
        line=dict(color=color, width=2, dash="dot"),
        fillcolor="rgba(0,0,0,0)",
    )


# -----------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------
st.title("EuroSAT Tile Classifier -- RGB ConvNeXtTiny")
st.caption(
    "Loads the trained RGB model, splits a large satellite scene into tiles, "
    "classifies every tile, and lets you click a tile to inspect its prediction."
)

if not os.path.exists(MODEL_PATH):
    st.error(
        f"Could not find `{MODEL_PATH}` next to app.py. "
        "Make sure the trained RGB model file is in the project folder."
    )
    st.stop()

labels = load_label_map(LABEL_MAP_PATH) if os.path.exists(LABEL_MAP_PATH) else DEFAULT_LABELS
model = load_classifier(MODEL_PATH)

with st.sidebar:
    st.header("Scene")
    image_file = st.file_uploader("Large RGB satellite image", type=["tif", "tiff", "jpg", "jpeg", "png"])

    st.header("Display")
    overlay_alpha = st.slider("Overlay opacity", 0.0, 1.0, 0.35, 0.05)

    st.header("Confidence Thresholds")
    high_thresh_pct = st.slider("High confidence >=", 50, 100, 90, 1)
    med_thresh_pct = st.slider("Medium confidence >=", 0, high_thresh_pct, 70, 1)
    high_thresh = high_thresh_pct / 100
    med_thresh = med_thresh_pct / 100

    st.header("Class Filter")
    st.caption("Only checked classes are highlighted on the map.")
    selected_classes = []
    for label in labels:
        if st.checkbox(label, value=True, key=f"filter_{label}"):
            selected_classes.append(label)

if not image_file:
    st.info("Upload a large RGB scene (.tif/.jpg/.png) in the sidebar to begin.")
    st.stop()

img = read_scene(image_file.getvalue(), image_file.name)  # (H, W, 3) uint8
h, w, _ = img.shape
n_rows_full, n_cols_full = h // TILE_SIZE, w // TILE_SIZE
h_crop, w_crop = n_rows_full * TILE_SIZE, n_cols_full * TILE_SIZE

if h_crop == 0 or w_crop == 0:
    st.error(f"Scene is smaller than one tile ({TILE_SIZE}x{TILE_SIZE} px).")
    st.stop()

img_cropped = img[:h_crop, :w_crop, :]
st.caption(
    f"Scene: {w} x {h} px -> cropped to {w_crop} x {h_crop} px "
    f"({n_rows_full} x {n_cols_full} tiles of {TILE_SIZE}px; "
    f"{w - w_crop}px / {h - h_crop}px trimmed from the right/bottom edge)."
)

if n_rows_full * n_cols_full > 4000:
    st.warning("This is a large scene -- classification may take a while, especially on CPU.")

results, n_rows, n_cols = classify_scene(model, img_cropped, TILE_SIZE, tuple(labels))

# Tag every tile with a confidence bucket, using the sidebar thresholds.
for res in results:
    res["conf_bucket"] = bucket_confidence(res["confidence"], high_thresh, med_thresh)

high_tiles = [r for r in results if r["conf_bucket"] == "High"]
med_tiles = [r for r in results if r["conf_bucket"] == "Medium"]
low_tiles = [r for r in results if r["conf_bucket"] == "Low"]
uncertain_tiles = med_tiles + low_tiles

# -----------------------------------------------------------------------
# Confidence / uncertainty analysis
# -----------------------------------------------------------------------
st.subheader("Confidence Analysis")
c1, c2, c3 = st.columns(3)
c1.metric("High confidence", f"{len(high_tiles)} tiles")
c2.metric("Medium confidence", f"{len(med_tiles)} tiles")
c3.metric("Low confidence", f"{len(low_tiles)} tiles")

show_uncertain = st.checkbox(
    f"Show uncertain regions on map ({len(uncertain_tiles)} tile(s) below {high_thresh_pct}% confidence)"
)

if uncertain_tiles:
    with st.expander(f"Inspect uncertain tiles ({len(uncertain_tiles)})"):
        sorted_uncertain = sorted(uncertain_tiles, key=lambda r: r["confidence"])
        options = [
            f"Row {r['row']}, Col {r['col']} -- {r['pred_label']} "
            f"({r['confidence'] * 100:.1f}%, {r['conf_bucket']})"
            for r in sorted_uncertain
        ]
        choice = st.selectbox("Uncertain tiles, lowest confidence first", options)
        chosen = sorted_uncertain[options.index(choice)]
        chosen_crop = img_cropped[
            chosen["row"] * TILE_SIZE:(chosen["row"] + 1) * TILE_SIZE,
            chosen["col"] * TILE_SIZE:(chosen["col"] + 1) * TILE_SIZE,
        ]
        uc1, uc2 = st.columns([1, 2])
        with uc1:
            st.image(chosen_crop, width=140)
        with uc2:
            st.write(f"**{chosen['pred_label']}** -- {chosen['confidence'] * 100:.1f}% confidence")
            probs_df = pd.DataFrame({"Class": labels, "Probability": chosen["probs"]}) \
                .sort_values("Probability", ascending=False)
            st.bar_chart(probs_df.set_index("Class"))

st.divider()

# -----------------------------------------------------------------------
# Map + tile inspector
# -----------------------------------------------------------------------
overlay_img = build_overlay_image(img_cropped, results, TILE_SIZE, overlay_alpha, set(selected_classes))

# Read the most recent click (if any) *before* building the figure, so the
# highlight rectangle can be drawn for the tile that's actually selected in
# this run -- Streamlit already updates session_state[key] by the time the
# script re-runs after a click, even before the chart is re-rendered below.
click_state = st.session_state.get("scene_map")
clicked_row, clicked_col = None, None
if click_state:
    points = (click_state.get("selection", {}) or {}).get("points")
    if points:
        clicked_col = int(points[0]["x"] // TILE_SIZE)
        clicked_row = int(points[0]["y"] // TILE_SIZE)

col_map, col_detail = st.columns([3, 1.3])

with col_detail:
    st.subheader("Legend")
    for label in labels:
        color = CLASS_COLORS.get(label, (255, 255, 255))
        dim = label not in selected_classes
        opacity = "0.35" if dim else "1"
        st.markdown(
            f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;opacity:{opacity};">'
            f'<div style="width:14px;height:14px;background:rgb{color};border-radius:3px;"></div>'
            f'<span style="font-size:0.9em;">{label}</span></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Selected Tile")
    # Keep the manual selectors and map selection synchronized.
    if "selected_row" not in st.session_state:
        st.session_state["selected_row"] = clicked_row or 0
    if "selected_col" not in st.session_state:
        st.session_state["selected_col"] = clicked_col or 0
    st.session_state["selected_row"] = min(st.session_state["selected_row"], n_rows - 1)
    st.session_state["selected_col"] = min(st.session_state["selected_col"], n_cols - 1)
    sel_row = st.number_input("Row", 0, n_rows - 1, key="selected_row")
    sel_col = st.number_input("Col", 0, n_cols - 1, key="selected_col")
    selected = next((r for r in results if r["row"] == sel_row and r["col"] == sel_col), None)

fig = px.imshow(overlay_img)
fig.data[0].hoverinfo = "skip"  # image trace is visual-only
fig.add_trace(build_tile_click_trace(results, n_rows, n_cols, TILE_SIZE))   # hover + click-to-select

shapes = make_grid_shapes(n_rows, n_cols, TILE_SIZE)

if show_uncertain:
    for r in med_tiles:
        shapes.append(make_uncertainty_shape(r["row"], r["col"], TILE_SIZE, "#FFA500"))
    for r in low_tiles:
        shapes.append(make_uncertainty_shape(r["row"], r["col"], TILE_SIZE, "#FF3B3B"))

if selected is not None:
    shapes.append(make_highlight_shape(sel_row, sel_col, TILE_SIZE))

fig.update_layout(
    shapes=shapes,
    margin=dict(l=0, r=0, t=0, b=0),
    # Streamlit receives Plotly selection events; Plotly needs this flag to
    # turn a single marker click into a point selection.
    clickmode="event+select",
    hovermode="closest",
    dragmode=False,
)
fig.update_xaxes(visible=False)
fig.update_yaxes(visible=False)

with col_map:
    st.plotly_chart(
        fig,
        use_container_width=True,
        on_select=sync_selected_tile,
        selection_mode="points",
        key="scene_map",
    )


with col_detail:
    if selected:
        st.metric(
            "Predicted class", selected["pred_label"],
            f"{selected['confidence'] * 100:.1f}% confidence ({selected['conf_bucket']})"
        )

        tile_crop = img_cropped[
            selected["row"] * TILE_SIZE:(selected["row"] + 1) * TILE_SIZE,
            selected["col"] * TILE_SIZE:(selected["col"] + 1) * TILE_SIZE,
        ]
        st.image(tile_crop, caption=f"Tile (row {selected['row']}, col {selected['col']})", width=160)

        probs_df = pd.DataFrame({"Class": labels, "Probability": selected["probs"]}) \
            .sort_values("Probability", ascending=False)
        st.bar_chart(probs_df.set_index("Class"))

st.divider()
st.subheader("All Tile Predictions")

results_df = pd.DataFrame(results)[["row", "col", "pred_label", "confidence", "conf_bucket"]]
if selected_classes:
    results_df = results_df[results_df["pred_label"].isin(selected_classes)]
st.dataframe(results_df, use_container_width=True, height=250)

st.download_button(
    "Download predictions as CSV",
    pd.DataFrame(results)[["row", "col", "pred_label", "confidence", "conf_bucket"]]
      .to_csv(index=False).encode("utf-8"),
    file_name="tile_predictions.csv",
    mime="text/csv",
)
