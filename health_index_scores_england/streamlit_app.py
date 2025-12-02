from __future__ import annotations

import json
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from ripser import ripser

import sys

try:  # Optional dependency for GeoJSON preprocessing
    from shapely.geometry import mapping, shape
    from shapely.ops import transform
except ImportError:
    mapping = shape = transform = None

try:
    from pyproj import Transformer
except ImportError:
    Transformer = None

try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        Image as PDFImage,
        Paragraph,
        SimpleDocTemplate,
        Spacer,
        Table,
        TableStyle,
    )

    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False

SHAPELY_AVAILABLE = all(v is not None for v in (mapping, shape, transform, Transformer))
TRANSFORMER = (
    Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)
    if SHAPELY_AVAILABLE
    else None
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))
sys.path.append(str(Path(__file__).resolve().parents[1] / ".tda_lib"))

import health_index_scores_england.ballmapper_utils as bm

DATA_PATH = Path(__file__).resolve().parent / "health_index_combined_2021.csv"
GEOJSON_CANDIDATES = list(
    Path(__file__).resolve().parent.glob("Local_Authority_Districts*.geojson")
)
GEOJSON_PATH = GEOJSON_CANDIDATES[0] if GEOJSON_CANDIDATES else None
SIMPLIFIED_GEOJSON_PATH = Path(__file__).resolve().parent / "england_la_simplified.geojson"
GEOJSON_SIMPLIFY_TOLERANCE = 500.0  # metres (data in British National Grid)
COORD_DECIMALS = 5
DEFAULT_FEATURES = [
    "Healthy People Domain",
    "Healthy Lives Domain",
    "Healthy Places Domain",
]
MAP_HOVER_FIELDS = [
    "Healthy People Domain",
    "Unemployment [Pl4]",
    "Life expectancy [Pe3]",
    "Mental health [Pe]",
]
MAP_PLOT_CONFIG = {
    "displaylogo": False,
    "scrollZoom": True,
    "modeBarButtonsToAdd": ["zoom2d", "zoomIn2d", "zoomOut2d", "resetScale2d"],
}
PRESET_CONFIGS = {
    "Custom": {
        "features": DEFAULT_FEATURES,
        "color": "Healthy People Domain",
        "notes": "Manually selected features/colour.",
    },
    "🧠 Mental health & deprivation": {
        "features": ["Child poverty [Pl4]", "Unemployment [Pl4]", "Mental health [Pe]"],
        "color": "Life expectancy [Pe3]",
        "notes": "Health expectancy differences driven by economic hardship + mental wellbeing.",
    },
    "🏃 Lifestyle gradient": {
        "features": ["Smoking [L1]", "Physical activity [L1]", "Healthy eating [L1]"],
        "color": "Healthy People Domain",
        "notes": "Behavioural risk clusters vs overall health.",
    },
    "🧠 Lifestyle Risk & Mental Wellbeing": {
        "features": [
            "Overweight and obesity in adults [L3]",
            "Alcohol misuse [L1]",
            "Drug misuse [L1]",
        ],
        "color": "Mental health [Pe]",
        "notes": "Topological link between substance use, obesity, and mental wellbeing.",
    },
    "🌳 Environment & access": {
        "features": ["Access to green space [Pl]", "Air pollution [Pl5]", "Distance to GP services [Pl2]"],
        "color": "Mental health [Pe]",
        "notes": "Urban–rural separation linked to mental health.",
    },
    "🧒 Education & youth wellbeing": {
        "features": ["Early years development [L2]", "Pupil attainment [L2]", "Teenage pregnancy [L2]"],
        "color": "Healthy Lives Domain",
        "notes": "Generational gradients through youth outcomes.",
    },
    "❤️ Preventive health": {
        "features": [
            "Cancer screening attendance [L4]",
            "Child vaccination coverage [L4]",
            "Overweight and obesity in adults [L3]",
        ],
        "color": "Life expectancy [Pe3]",
        "notes": "Preventive engagement vs longevity.",
    },
    "🧬 Chronic Disease Burden": {
        "features": [
            "Cancer [Pe5]",
            "Cardiovascular conditions [Pe5]",
            "Diabetes [Pe5]",
            "Respiratory conditions [Pe5]",
            "Dementia [Pe5]",
        ],
        "color": "Healthy People Domain",
        "notes": "Overlapping chronic disease patterns and their effect on overall health.",
    },
    "🌆 Urban Environment & Air Quality": {
        "features": [
            "Air pollution [Pl5]",
            "Noise complaints [Pl5]",
            "Household overcrowding [Pl5]",
            "Access to green space [Pl]",
            "Road safety [Pl5]",
        ],
        "color": "Healthy Places Domain",
        "notes": "Urban environmental stressors vs metropolitan/rural health environments.",
    },
    "🔄 Cross-domain summary": {
        "features": [
            "Healthy People Domain",
            "Healthy Lives Domain",
            "Healthy Places Domain",
        ],
        "color": "Overweight and obesity in adults [L3]",
        "notes": "Macro-level topology of national health balance.",
    },
    "⏳ Social Gradient of Longevity": {
        "features": [
            "Child poverty [Pl4]",
            "Unemployment [Pl4]",
            "Job-related training [Pl4]",
            "Physical activity [L1]",
            "Healthy eating [L1]",
            "Smoking [L1]",
            "Alcohol misuse [L1]",
            "Cancer screening attendance [L4]",
            "Child vaccination coverage [L4]",
            "Access to green space [Pl]",
            "Air pollution [Pl5]",
        ],
        "color": "Life expectancy [Pe3]",
        "notes": "Socioeconomic–behavioural–environmental gradient vs life expectancy.",
    },
    "🌐 Regional Health Inequality": {
        "features": [
            "Life expectancy [Pe3]",
            "Mental health [Pe]",
            "Diabetes [Pe5]",
            "Respiratory conditions [Pe5]",
            "Physical activity [L1]",
            "Smoking [L1]",
            "Alcohol misuse [L1]",
            "Access to green space [Pl]",
            "Air pollution [Pl5]",
            "Unemployment [Pl4]",
        ],
        "color": "Healthy People Domain",
        "notes": (
            "Chronic disease, behaviours, environment, and deprivation jointly shaping health inequality. "
            "Suggested ε≈0.4."
        ),
    },
    "🌱 Healthy Places Mega-Structure": {
        "features": [
            # Access to services
            "Access to green space [Pl]", "Private outdoor space [Pl1]",
            "Access to services [Pl]", "Distance to GP services [Pl2]",
            "Distance to pharmacies [Pl2]", "Distance to sports or leisure facilities [Pl2]",
            "Internet access [Pl2]", "Patients offered acceptable GP practice appointments [Pl2]",

            # Crime / safety
            "Low-level crime [Pl3]", "Personal crime [Pl3]", "Road safety [Pl5]",

            # Living conditions
            "Noise complaints [Pl5]", "Air pollution [Pl5]", "Household overcrowding [Pl5]",
            "Rough sleeping [Pl5]",

            # Socioeconomic
            "Child poverty [Pl4]", "Job-related training [Pl4]", "Unemployment [Pl4]"
        ],
        "color": "Avoidable mortality [Pe3]",
        "notes": "Pure environment/deprivation manifold. Often reveals strong topological gradient lines for mortality levels."
    },
    "🌐 Digital Access & Social Participation": {
        "features": [
            "Internet access [Pl2]",
            "Distance to sports or leisure facilities [Pl2]",
            "Distance to GP services [Pl2]",
            "Patients offered acceptable GP practice appointments [Pl2]",
            "Young people in education, employment and apprenticeships [L2]",
            "Job-related training [Pl4]",
            "Pupil absences [L2]"
        ],
        "color": "Avoidable mortality [Pe3]",
        "notes": "Tests whether digital connectivity, access, and youth engagement create unexpected mortality patterns."
    },
}

HEATMAP_CATEGORY_BUNDLES = {
    "Mental health & wellbeing": [
        "Mental health [Pe]",
        "Children's social, emotional and mental health [Pe2]",
        "Mental health conditions [Pe2]",
        "Self-harm [Pe2]",
        "Suicides [Pe2]",
        "Personal well-being [Pe]",
        "Feelings of anxiety [Pe4]",
        "Happiness [Pe4]",
        "Life satisfaction [Pe4]",
    ],
    "Physical health burden": [
        "Physical health conditions [Pe]",
        "Cancer [Pe5]",
        "Cardiovascular conditions [Pe5]",
        "Diabetes [Pe5]",
        "Respiratory conditions [Pe5]",
        "Musculoskeletal conditions [Pe5]",
        "Kidney and liver disease [Pe5]",
        "Dementia [Pe5]",
    ],
    "Wealth, work & deprivation": [
        "Child poverty [Pl4]",
        "Unemployment [Pl4]",
        "Job-related training [Pl4]",
        "Workplace safety [Pl4]",
        "Rough sleeping [Pl5]",
        "Household overcrowding [Pl5]",
    ],
    "Healthy behaviours & risks": [
        "Smoking [L1]",
        "Alcohol misuse [L1]",
        "Drug misuse [L1]",
        "Healthy eating [L1]",
        "Physical activity [L1]",
        "Sedentary behaviour [L1]",
        "Overweight and obesity in adults [L3]",
        "Overweight and obesity in children [L3]",
    ],
    "Preventive care & access": [
        "Cancer screening attendance [L4]",
        "Child vaccination coverage [L4]",
        "Access to services [Pl]",
        "Distance to GP services [Pl2]",
        "Distance to pharmacies [Pl2]",
        "Distance to sports or leisure facilities [Pl2]",
        "Internet access [Pl2]",
        "Patients offered acceptable GP practice appointments [Pl2]",
    ],
    "Environment & place": [
        "Healthy Places Domain",
        "Access to green space [Pl]",
        "Private outdoor space [Pl1]",
        "Air pollution [Pl5]",
        "Noise complaints [Pl5]",
        "Road safety [Pl5]",
    ],
}


YEAR_OPTIONS = sorted(bm.available_years().keys(), reverse=True)


@st.cache_data(show_spinner=False)
def load_dataset(year: str) -> pd.DataFrame:
    return bm.load_health_index_dataset(year=year)


def _round_coords(value: Any, decimals: int) -> Any:
    if isinstance(value, (float, int)):
        return round(float(value), decimals)
    if isinstance(value, (list, tuple)):
        return [_round_coords(item, decimals) for item in value]
    return value


def _preprocess_geojson(source_path: Path) -> Dict[str, Any]:
    with source_path.open("r", encoding="utf-8") as fh:
        raw_data = json.load(fh)
    processed_features: List[Dict[str, Any]] = []
    for feature in raw_data.get("features", []):
        props = feature.get("properties", {}) or {}
        lad_code = str(props.get("LAD21CD", "")).strip()
        if not lad_code.startswith("E"):
            continue
        geometry = feature.get("geometry")
        if geometry and SHAPELY_AVAILABLE:
            try:
                geom = shape(geometry)
                if not geom.is_valid:
                    geom = geom.buffer(0)
                geom = geom.simplify(GEOJSON_SIMPLIFY_TOLERANCE, preserve_topology=True)
                if TRANSFORMER is not None:
                    geom = transform(TRANSFORMER.transform, geom)
                geometry = mapping(geom)
            except Exception:
                geometry = feature.get("geometry")
        if geometry:
            geometry = {
                "type": geometry.get("type"),
                "coordinates": _round_coords(geometry.get("coordinates"), COORD_DECIMALS),
            }
        processed_features.append(
            {
                "type": feature.get("type", "Feature"),
                "properties": {
                    "LAD21CD": lad_code,
                    "LAD21NM": props.get("LAD21NM", ""),
                },
                "geometry": geometry,
            }
        )
    return {"type": raw_data.get("type", "FeatureCollection"), "features": processed_features}


def _ensure_simplified_geojson(base_path: Path | None) -> Tuple[Dict[str, Any], Path]:
    if SIMPLIFIED_GEOJSON_PATH.exists():
        with SIMPLIFIED_GEOJSON_PATH.open("r", encoding="utf-8") as fh:
            return json.load(fh), SIMPLIFIED_GEOJSON_PATH
    if base_path is None or not base_path.exists():
        raise FileNotFoundError(
            "Local authority GeoJSON file not found. Place it in `health_index_scores_england/`."
        )
    if SHAPELY_AVAILABLE:
        data = _preprocess_geojson(base_path)
        SIMPLIFIED_GEOJSON_PATH.parent.mkdir(parents=True, exist_ok=True)
        with SIMPLIFIED_GEOJSON_PATH.open("w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
        return data, SIMPLIFIED_GEOJSON_PATH
    with base_path.open("r", encoding="utf-8") as fh:
        return json.load(fh), base_path


@st.cache_data(show_spinner=False)
def load_geojson(path: str | None = None) -> Tuple[Dict[str, Any], List[str]]:
    base_path = Path(path) if path else GEOJSON_PATH
    data, _ = _ensure_simplified_geojson(base_path)
    codes = sorted(
        {
            feature.get("properties", {}).get("LAD21CD")
            for feature in data.get("features", [])
            if feature.get("properties", {}).get("LAD21CD")
        }
    )
    return data, codes


def sidebar_filters(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, List[str], List[str], bool]:
    st.sidebar.markdown("### Filters")
    source_options = sorted(df["Source"].unique())
    source_state_key = "filters_sources"
    stored_sources = st.session_state.get(source_state_key, source_options)
    default_sources = [s for s in stored_sources if s in source_options] or source_options
    selected_sources = st.sidebar.multiselect("Source", source_options, default=default_sources)
    if not selected_sources:
        selected_sources = source_options
    sources_changed = set(selected_sources) != set(stored_sources)
    st.session_state[source_state_key] = selected_sources
    area_types = sorted(df["Area Type"].unique())
    default_area_types = [atype for atype in area_types if atype.lower() != "country"]
    area_state_key = "filters_area_types"
    stored_area_types = st.session_state.get(area_state_key, default_area_types)
    default_selected_area = [a for a in stored_area_types if a in area_types] or default_area_types or area_types
    selected_area_types = st.sidebar.multiselect("Area Type", area_types, default=default_selected_area)
    if not selected_area_types:
        selected_area_types = default_selected_area
    areas_changed = set(selected_area_types) != set(stored_area_types)
    st.session_state[area_state_key] = selected_area_types
    filtered = df[df["Source"].isin(selected_sources) & df["Area Type"].isin(selected_area_types)].copy()
    filters_changed = sources_changed or areas_changed
    return filtered, selected_sources, selected_area_types, filters_changed


def choose_features(df: pd.DataFrame) -> tuple[List[str], str, bool, bool]:
    st.sidebar.markdown("### Feature Selection")
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"Source", "Year", "Area Code", "Area Name", "Area Type"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    default_features = [f for f in DEFAULT_FEATURES if f in numeric_cols][:4] or numeric_cols[:4]
    stored_manual_features = st.session_state.get("manual_feature_cols", default_features)
    default_selection = [f for f in stored_manual_features if f in numeric_cols] or default_features
    feature_cols = st.sidebar.multiselect(
        "Ball Mapper features", numeric_cols, default=default_selection
    )
    if not feature_cols:
        feature_cols = default_selection
    manual_features_changed = set(feature_cols) != set(stored_manual_features)
    st.session_state["manual_feature_cols"] = feature_cols
    norm_options = ["minmax", "zscore"]
    stored_norm = st.session_state.get("norm_method_state", norm_options[0])
    if stored_norm not in norm_options:
        stored_norm = norm_options[0]
    norm_method = st.sidebar.selectbox(
        "Normalization method", norm_options, index=norm_options.index(stored_norm)
    )
    norm_changed = norm_method != stored_norm
    st.session_state["norm_method_state"] = norm_method
    st.sidebar.caption("Tip: When selecting many features, try **z-score** normalisation and a larger ε range to maintain connectivity.")
    return feature_cols or default_features, norm_method, manual_features_changed, norm_changed


def render_dataset_summary(df: pd.DataFrame, year: str) -> None:
    st.subheader("Filtered dataset")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", len(df))
    c2.metric("Sources", df["Source"].nunique())
    c3.metric("Area types", df["Area Type"].nunique())
    c4.metric("Year", year)
    with st.expander("Preview filtered dataset", expanded=False):
        st.dataframe(df)
        st.download_button(
            "Download filtered CSV",
            data=df.to_csv(index=False),
            file_name="health_index_filtered.csv",
            mime="text/csv",
        )


def _aggregate_bundle_series(
    df: pd.DataFrame,
    columns: List[str],
    method: str = "mean",
) -> pd.Series:
    numeric_subset = df[columns].apply(pd.to_numeric, errors="coerce")
    if method == "median":
        return numeric_subset.median(axis=1, skipna=True)
    return numeric_subset.mean(axis=1, skipna=True)


def build_correlation_dataset(
    df: pd.DataFrame,
    *,
    bundles: List[str],
    metrics: List[str],
    agg_method: str,
) -> tuple[pd.DataFrame, Dict[str, List[str]]]:
    data: Dict[str, pd.Series] = {}
    missing_columns: Dict[str, List[str]] = {}
    for bundle in bundles:
        cols = [col for col in HEATMAP_CATEGORY_BUNDLES.get(bundle, []) if col in df.columns]
        if not cols:
            missing_columns[bundle] = HEATMAP_CATEGORY_BUNDLES.get(bundle, [])
            continue
        data[bundle] = _aggregate_bundle_series(df, cols, method=agg_method)
    for metric in metrics:
        if metric in df.columns:
            data[metric] = pd.to_numeric(df[metric], errors="coerce")
    matrix = pd.DataFrame(data)
    matrix = matrix.dropna(axis=1, how="all")
    return matrix, missing_columns


def render_correlation_explorer(df: pd.DataFrame) -> None:
    st.header("Correlation explorer")
    st.caption(
        "Add or remove indicator bundles to gauge how domains (e.g. mental health vs wealth) co-move. "
        "Use Spearman when dealing with skewed scores."
    )
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"Source", "Year", "Area Code", "Area Name", "Area Type"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    if len(numeric_cols) < 2:
        st.info("Need at least two numeric indicators to compute correlations.")
        return
    available_bundles = [
        bundle
        for bundle, cols in HEATMAP_CATEGORY_BUNDLES.items()
        if any(col in numeric_cols for col in cols)
    ]
    default_bundles = [
        bundle
        for bundle in ["Mental health & wellbeing", "Physical health burden", "Wealth, work & deprivation"]
        if bundle in available_bundles
    ]
    with st.expander("Correlation heatmap", expanded=True):
        selected_bundles = st.multiselect(
            "Category bundles",
            options=available_bundles,
            default=default_bundles,
            key="corr_bundle_options",
        )
        manual_default = [f for f in DEFAULT_FEATURES if f in numeric_cols]
        selected_metrics = st.multiselect(
            "Individual indicators",
            options=sorted(numeric_cols),
            default=manual_default,
            key="corr_manual_metrics",
        )
        agg_method = st.selectbox(
            "Bundle aggregation",
            options=["mean", "median"],
            format_func=lambda x: "Median" if x == "median" else "Mean",
            key="corr_agg_method",
        )
        corr_method = st.selectbox(
            "Correlation method",
            options=["pearson", "spearman"],
            format_func=lambda x: x.title(),
            key="corr_method",
        )
        if not selected_bundles and not selected_metrics:
            st.info("Pick at least one bundle or indicator to build the heatmap.")
            return
        corr_data, missing = build_correlation_dataset(
            df,
            bundles=selected_bundles,
            metrics=selected_metrics,
            agg_method=agg_method,
        )
        if corr_data.shape[1] < 2:
            st.info("Need at least two valid series after filtering. Try adding more bundles or indicators.")
            return
        corr_matrix = corr_data.corr(method=corr_method).round(3)
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
            aspect="auto",
        )
        fig.update_layout(
            height=520,
            margin=dict(l=10, r=10, t=40, b=10),
            coloraxis_colorbar=dict(title=f"{corr_method.title()} r"),
        )
        st.plotly_chart(fig, use_container_width=True)
        min_abs = st.slider(
            "Highlight pairs with |corr| ≥",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.05,
            key="corr_threshold",
        )
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
        triangular = corr_matrix.where(mask)
        corr_pairs = (
            triangular.stack()
            .rename("correlation")
            .reset_index()
            .rename(columns={"level_0": "feature_a", "level_1": "feature_b"})
        )
        if not corr_pairs.empty:
            corr_pairs["abs_corr"] = corr_pairs["correlation"].abs()
            highlights = corr_pairs[corr_pairs["abs_corr"] >= min_abs].sort_values("abs_corr", ascending=False)
            st.markdown("#### Notable relationships")
            if highlights.empty:
                st.write("No pairs exceed the selected threshold.")
            else:
                st.dataframe(highlights, use_container_width=True)
            st.download_button(
                "Download correlation matrix (CSV)",
                data=corr_matrix.to_csv(),
                file_name="health_index_correlation_matrix.csv",
                mime="text/csv",
            )
        if missing:
            missing_text = ", ".join(sorted(missing.keys()))
            st.caption(
                f"Bundles skipped (no matching columns in current filter): {missing_text}. "
                "Switch year/filters if you need them."
            )


def build_area_assignment_df(
    df: pd.DataFrame,
    nodes: List[bm.BallMapperNode],
    color_metric: str,
    feature_cols: List[str],
) -> pd.DataFrame:
    records: List[Dict[str, Any]] = []
    metric_columns = set(MAP_HOVER_FIELDS + feature_cols + [color_metric])
    for node in nodes:
        ball_label = f"Ball {node.index}"
        for member_idx in node.members:
            row = df.iloc[member_idx]
            area_code = row.get("Area Code")
            if not isinstance(area_code, str):
                continue
            record: Dict[str, Any] = {
                "Area Code": area_code.strip(),
                "Area Name": row.get("Area Name", ""),
                "Area Type": row.get("Area Type", ""),
                "Node": node.index,
                "ball_label": ball_label,
            }
            for col in metric_columns:
                value = node.metrics.get(col)
                if value is None or (isinstance(value, float) and np.isnan(value)):
                    value = row.get(col)
                record[col] = value
            records.append(record)
    map_df = pd.DataFrame(records)
    if map_df.empty:
        return map_df
    map_df = map_df.dropna(subset=["Area Code"])
    map_df = map_df.drop_duplicates(subset=["Area Code"], keep="first")
    map_df["Area Code"] = map_df["Area Code"].astype(str)
    for col in metric_columns:
        if col in map_df.columns:
            map_df[col] = pd.to_numeric(map_df[col], errors="coerce")
    map_df["ball_label"] = map_df["ball_label"].astype(str)
    return map_df


def render_geo_overlay(
    map_df: pd.DataFrame,
    geojson: Dict[str, Any],
    color_metric: str,
    feature_cols: List[str],
) -> Optional[go.Figure]:
    if map_df.empty:
        st.info("No LTLA/UTLA local authority rows available for the geographic overlay. Adjust filters.")
        return None
    if color_metric not in map_df.columns:
        st.warning(f"{color_metric} unavailable for the geographic overlay.")
        return None
    color_values = pd.to_numeric(map_df[color_metric], errors="coerce")
    map_df = map_df.assign(**{color_metric: color_values})
    if color_values.notna().sum() == 0:
        st.warning(f"{color_metric} has no numeric data to colour the map.")
        return None
    hover_fields = []
    if color_metric in map_df.columns:
        hover_fields.append(color_metric)
    for feat in feature_cols:
        if feat in map_df.columns:
            hover_fields.append(feat)
    hover_fields.extend(col for col in MAP_HOVER_FIELDS if col in map_df.columns)
    hover_fields = list(dict.fromkeys(hover_fields))
    hover_data = {col: True for col in hover_fields}
    hover_data["Node"] = True
    hover_data["ball_label"] = False
    color_min = float(color_values.min())
    color_max = float(color_values.max())
    fig = px.choropleth_mapbox(
        map_df,
        geojson=geojson,
        locations="Area Code",
        featureidkey="properties.LAD21CD",
        color=color_metric,
        hover_name="Area Name",
        hover_data=hover_data,
        mapbox_style="carto-positron",
        zoom=5.0,
        opacity=0.8,
        center={"lat": 53.4, "lon": -1.6},
        color_continuous_scale="Viridis",
        range_color=(color_min, color_max),
    )
    fig.update_layout(
        margin=dict(l=10, r=10, t=40, b=10),
        coloraxis_colorbar=dict(title=color_metric),
        legend_title_text="",
        height=720,
    )
    st.plotly_chart(fig, use_container_width=True, config=MAP_PLOT_CONFIG)
    return fig


def render_barcodes(barcodes: Dict[str, np.ndarray]) -> go.Figure | None:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, subplot_titles=("H0 barcode", "H1 barcode"))
    colors = {"H0": "skyblue", "H1": "salmon"}
    for row, dim in enumerate(["H0", "H1"], start=1):
        diagram = barcodes.get(dim, np.empty((0, 2)))
        for idx, (birth, death) in enumerate(diagram):
            death = death if np.isfinite(death) else birth + 1.0
            fig.add_trace(
                go.Scatter(
                    x=[birth, death],
                    y=[idx, idx],
                    mode="lines",
                    line=dict(color=colors[dim], width=3),
                    showlegend=False,
                ),
                row=row,
                col=1,
            )
    fig.update_yaxes(title_text="Intervals", showticklabels=False, row=1, col=1)
    fig.update_xaxes(title_text="Filtration value")
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=40, b=20),
        plot_bgcolor="rgba(12,12,12,1)",
        paper_bgcolor="rgba(12,12,12,1)",
        font=dict(color="#f0f0f0"),
    )
    st.plotly_chart(fig, use_container_width=True)
    return fig


def summarize_betti_numbers(barcodes: Dict[str, np.ndarray]) -> Dict[str, int]:
    betti: Dict[str, int] = {}
    for dim, diagram in barcodes.items():
        dim_label = dim[-1] if dim.startswith("H") else dim
        betti[f"β{dim_label}"] = int(len(diagram))
    return betti


def figure_to_png_bytes(fig: Optional[go.Figure], scale: int = 3) -> Optional[bytes]:
    if fig is None:
        return None
    try:
        return fig.to_image(format="png", scale=scale)
    except Exception:
        return None


def build_report_context(
    *,
    data_year: str,
    preset_name: str,
    preset_notes: str,
    epsilon: float,
    normalization: str,
    feature_cols: List[str],
    color_metric: str,
    size_metric: str,
    sources: List[str],
    area_types: List[str],
    filtered_df: pd.DataFrame,
    node_df: pd.DataFrame,
    node_summary: str,
    ballmapper_fig: go.Figure,
    map_fig: Optional[go.Figure],
    barcode_fig: Optional[go.Figure],
    ballmapper_png_bytes: Optional[bytes],
) -> Dict[str, Any]:
    dataset_stats = {
        "rows": int(len(filtered_df)),
        "sources": len(sources),
        "area_types": len(area_types),
        "year": data_year,
    }
    node_stats = {
        "count": int(len(node_df)),
        "avg_size": float(node_df["size"].mean()) if not node_df.empty else 0.0,
    }
    top_nodes: List[Dict[str, Any]] = []
    bottom_nodes: List[Dict[str, Any]] = []
    if not node_df.empty and color_metric in node_df.columns:
        top_nodes = node_df.sort_values(color_metric, ascending=False).head(5).to_dict("records")
        bottom_nodes = node_df.sort_values(color_metric, ascending=True).head(5).to_dict("records")
    ball_img = figure_to_png_bytes(ballmapper_fig, scale=4) if ballmapper_fig else None
    images = {
        "ballmapper": ball_img or ballmapper_png_bytes,
        "map": figure_to_png_bytes(map_fig, scale=3),
        "barcode": figure_to_png_bytes(barcode_fig, scale=3),
    }
    context = {
        "title": "Health Index Ball Mapper Report",
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "preset_name": preset_name,
        "preset_notes": preset_notes,
        "epsilon": f"{epsilon:.3f}",
        "normalization": normalization,
        "feature_cols": feature_cols,
        "color_metric": color_metric,
        "size_metric": size_metric,
        "sources": sources,
        "area_types": area_types,
        "dataset": dataset_stats,
        "node_summary": node_summary,
        "node_stats": node_stats,
        "top_nodes": top_nodes,
        "bottom_nodes": bottom_nodes,
        "images": images,
        "year": data_year,
    }
    return context


def render_report_pdf(context: Dict[str, Any]) -> bytes:
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab is not installed; unable to generate PDF.")
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4, title=context["title"])
    styles = getSampleStyleSheet()
    story: List[Any] = []

    def add_paragraph(text: str, style_name: str = "Normal", space: float = 10) -> None:
        story.append(Paragraph(text, styles[style_name]))
        story.append(Spacer(1, space))

    add_paragraph(f"<b>{context['title']}</b>", "Title", space=6)
    add_paragraph(
        f"Generated: {context['generated_at']} &nbsp;&nbsp; Preset: <b>{context['preset_name']}</b> &nbsp;&nbsp; ε={context['epsilon']}",
        space=12,
    )
    config_lines = [
        f"Normalisation: {context['normalization']}",
        f"Colour metric: {context['color_metric']}",
        f"Node size metric: {context['size_metric']}",
        f"Features ({len(context['feature_cols'])}): {', '.join(context['feature_cols'])}",
    ]
    if context["preset_notes"]:
        config_lines.append(f"Preset notes: {context['preset_notes']}")
    add_paragraph("<br/>".join(config_lines))

    dataset = context["dataset"]
    node_stats = context["node_stats"]
    add_paragraph(
        f"Year: {dataset['year']} &nbsp;&nbsp; Rows: {dataset['rows']} &nbsp;&nbsp; Sources: {dataset['sources']} &nbsp;&nbsp; Area types: {dataset['area_types']} "
        f"&nbsp;&nbsp; Nodes: {node_stats['count']} &nbsp;&nbsp; Avg node size: {node_stats['avg_size']:.1f}"
    )
    if context["node_summary"]:
        add_paragraph(f"<b>Key observations:</b> {context['node_summary']}")

    def add_image_section(label: str, image_bytes: Optional[bytes]) -> None:
        if not image_bytes:
            return
        story.append(Paragraph(f"<b>{label}</b>", styles["Heading2"]))
        img_stream = BytesIO(image_bytes)
        image_reader = ImageReader(img_stream)
        width, height = image_reader.getSize()
        max_width = 6.2 * inch
        aspect = height / width if width else 1.0
        img_stream.seek(0)
        story.append(PDFImage(img_stream, width=max_width, height=max_width * aspect))
        story.append(Spacer(1, 12))

    add_image_section("Ball Mapper graph", context["images"].get("ballmapper"))
    add_image_section("Geographic overlay", context["images"].get("map"))
    add_image_section("Persistence barcodes", context["images"].get("barcode"))

    def add_table(title: str, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        story.append(Paragraph(f"<b>{title}</b>", styles["Heading2"]))
        data = [["Label", "Size", context["color_metric"]]]
        for row in rows:
            data.append(
                [
                    row.get("label", ""),
                    str(row.get("size", "")),
                    f"{row.get(context['color_metric'], 0):.2f}",
                ]
            )
        table = Table(data, colWidths=[3.5 * inch, 1.0 * inch, 1.5 * inch])
        table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f0f0f0")),
                    ("ALIGN", (1, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ]
            )
        )
        story.append(table)
        story.append(Spacer(1, 12))

    add_table(f"Top nodes by {context['color_metric']}", context["top_nodes"])
    add_table(f"Lowest nodes by {context['color_metric']}", context["bottom_nodes"])

    doc.build(story)
    buffer.seek(0)
    return buffer.getvalue()


def build_plotly_graph(
    G: nx.Graph,
    color_metric: str,
    size_metric: str,
    feature_cols: List[str],
    background: str = "dark",
) -> go.Figure:
    pos = bm.compute_layout(G)
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    colors = [G.nodes[n][color_metric] for n in G.nodes]
    counts = np.array([G.nodes[n].get(size_metric, G.nodes[n]["size"]) for n in G.nodes], dtype=float)
    if len(counts) == 0:
        node_sizes = []
    else:
        scaled_counts = np.cbrt(np.clip(counts, a_min=0, a_max=None))
        min_px, max_px = 20, 120
        if scaled_counts.max() == scaled_counts.min():
            node_sizes = [((min_px + max_px) / 2) for _ in counts]
        else:
            node_sizes = list(
                np.interp(
                    scaled_counts,
                    [float(scaled_counts.min()), float(scaled_counts.max())],
                    [min_px, max_px],
                )
            )
    hovertext = []
    for n in G.nodes:
        base_lines = [
            f"Node {n}",
            f"Label: {G.nodes[n]['label']}",
            f"Area type: {G.nodes[n].get('area_type', '—')}",
            f"Source: {G.nodes[n].get('source', '—')}",
            f"Size: {int(G.nodes[n]['size'])}",
            f"{color_metric}: {G.nodes[n].get(color_metric, 0):.2f}",
        ]
        feature_lines = [
            f"{feat}: {G.nodes[n].get(feat, 0):.2f}"
            for feat in feature_cols
            if feat in G.nodes[n]
        ]
        hovertext.append("<br>".join(base_lines + feature_lines))
    edge_color = "rgba(60,60,60,0.6)" if background == "light" else "rgba(200,200,200,0.4)"
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color=edge_color),
        hoverinfo="none",
        mode="lines",
    )
    text_color = "#111111" if background == "light" else "#f0f0f0"
    node_trace = go.Scatter(
        x=[pos[n][0] for n in G.nodes],
        y=[pos[n][1] for n in G.nodes],
        mode="markers+text",
        text=[str(n) for n in G.nodes],
        textposition="middle center",
        textfont=dict(size=10, color=text_color),
        marker=dict(
            size=node_sizes,
            color=colors,
            colorscale="Viridis",
            showscale=True,
            colorbar=dict(title=color_metric.replace("_", " ").title()),
            line=dict(color="#333333", width=1),
        ),
        hoverinfo="text",
        hovertext=hovertext,
    )
    fig = go.Figure(data=[edge_trace, node_trace])
    bg_color = "rgba(12,12,12,1)" if background == "dark" else "#ffffff"
    font_color = "#f0f0f0" if background == "dark" else "#111111"
    fig.update_layout(
        title=f"Ball Mapper Graph (color={color_metric})",
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=720,
        margin=dict(l=20, r=20, t=60, b=110),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=font_color),
    )
    return fig


def compute_barcodes(points: np.ndarray, maxdim: int = 1) -> Dict[str, np.ndarray]:
    result = ripser(points, maxdim=maxdim)
    diagrams = result["dgms"]
    return {"H0": diagrams[0], "H1": diagrams[1] if len(diagrams) > 1 else np.empty((0, 2))}


def summarize_nodes(node_df: pd.DataFrame, color_metric: str) -> str:
    if node_df.empty or color_metric not in node_df.columns:
        return "No nodes to summarise."
    largest = node_df.sort_values("size", ascending=False).iloc[0]
    max_color = node_df.sort_values(color_metric, ascending=False).iloc[0]
    min_color = node_df.sort_values(color_metric, ascending=True).iloc[0]
    return (
        f"Largest ball: **{largest['label']}** (covers {int(largest['size'])} records). "
        f"Highest {color_metric}: **{max_color['label']}** ({max_color[color_metric]:.2f}). "
        f"Lowest {color_metric}: **{min_color['label']}** ({min_color[color_metric]:.2f})."
    )


def main() -> None:
    st.set_page_config(page_title="Health Index Ball Mapper", layout="wide")
    st.title("Health Index Ball Mapper · England LTLA (2015–2021)")
    st.markdown(
        "Interactively explore LTLA-level Health Index scores across 2015–2021. "
        "Lock presets, auto-tune ε, inspect Ball Mapper topology, geographic overlays, persistence barcodes, "
        "and export PDF summaries for any selected year. Data source: [ONS Health Index scores for England]"
        "(https://www.ons.gov.uk/peoplepopulationandcommunity/healthandsocialcare/healthandwellbeing/datasets/healthindexscoresengland)."
    )
    if not YEAR_OPTIONS:
        st.error("No health index datasets found.")
        return
    selected_year = st.sidebar.selectbox("Data year", YEAR_OPTIONS, index=0, key="data_year")
    previous_year = st.session_state.get("selected_year_prev")
    year_changed = previous_year is not None and previous_year != selected_year
    st.session_state["selected_year_prev"] = selected_year
    if "has_generated_plot" not in st.session_state:
        st.session_state["has_generated_plot"] = False
    df = load_dataset(selected_year)
    filtered_df, selected_sources, selected_area_types, filters_changed = sidebar_filters(df)
    if "bm_feature_cols" not in st.session_state:
        st.session_state["bm_feature_cols"] = DEFAULT_FEATURES
    if "bm_norm_method" not in st.session_state:
        st.session_state["bm_norm_method"] = "minmax"
    previous_final_features = st.session_state.get("bm_feature_cols", DEFAULT_FEATURES)
    feature_cols, norm_method, manual_features_changed, norm_changed = choose_features(filtered_df)
    st.session_state["bm_norm_method"] = norm_method
    if filtered_df.empty:
        st.warning("No data after filtering. Adjust sidebar filters.")
        return
    render_dataset_summary(filtered_df, selected_year)
    analysis_view = st.sidebar.radio(
        "Analysis view",
        options=["Ball Mapper", "Correlation explorer"],
        index=0,
        key="analysis_view_selector",
    )
    if analysis_view == "Correlation explorer":
        render_correlation_explorer(filtered_df)
        return

    st.header("Ball Mapper Playground")
    range_col, toggle_col = st.columns([0.8, 0.2])
    with toggle_col:
        wide_range = st.checkbox("ε up to 10.0", value=False, key="epsilon_wide")
    max_eps = 10.0 if wide_range else 0.5
    if "epsilon_value" not in st.session_state:
        st.session_state["epsilon_value"] = 0.2
    current_epsilon = min(st.session_state["epsilon_value"], max_eps)
    with range_col:
        epsilon = st.slider(
            "Ball radius ε",
            min_value=0.05,
            max_value=max_eps,
            value=current_epsilon,
            step=0.01,
        )
    epsilon = min(max(epsilon, 0.05), max_eps)
    epsilon_changed = abs(epsilon - st.session_state["epsilon_value"]) > 1e-9
    st.session_state["epsilon_value"] = epsilon
    numeric_cols = [
        col
        for col in filtered_df.columns
        if col not in {"Source", "Year", "Area Code", "Area Name", "Area Type"}
        and pd.api.types.is_numeric_dtype(filtered_df[col])
    ]
    if not numeric_cols:
        st.warning("No numeric columns available to colour by.")
        return
    default_color = "Healthy People Domain" if "Healthy People Domain" in numeric_cols else numeric_cols[0]
    color_metric = default_color
    if "color_metric_value" not in st.session_state:
        st.session_state["color_metric_value"] = default_color
    preset_name = st.selectbox("Preset theme", list(PRESET_CONFIGS.keys()), index=0)
    preset_cfg = PRESET_CONFIGS[preset_name]
    if preset_name != "Custom":
        preset_features = [f for f in preset_cfg["features"] if f in numeric_cols]
        if preset_features:
            feature_cols = preset_features
            st.session_state["bm_feature_cols"] = feature_cols
        if preset_cfg["color"] in numeric_cols:
            color_metric = preset_cfg["color"]
        st.caption(f"{preset_cfg['notes']}  \nFeatures: {', '.join(feature_cols)} | Colour: {color_metric}")
    feature_cols = feature_cols or DEFAULT_FEATURES
    features_changed = manual_features_changed or (set(feature_cols) != set(previous_final_features))
    st.session_state["bm_feature_cols"] = feature_cols
    current_color_value = st.session_state.get("color_metric_value", color_metric)
    if current_color_value not in numeric_cols:
        current_color_value = color_metric
    color_metric = st.selectbox(
        "Colour metric",
        numeric_cols,
        index=numeric_cols.index(current_color_value),
    )
    st.session_state["color_metric_value"] = color_metric
    size_state_key = "size_metric_value"
    size_options = ["size"] + feature_cols
    current_size_value = st.session_state.get(size_state_key, "size")
    if current_size_value not in size_options:
        current_size_value = "size"
    size_metric = st.selectbox(
        "Node size metric (display only)",
        size_options,
        index=size_options.index(current_size_value),
    )
    st.session_state[size_state_key] = size_metric

    background_toggle = st.checkbox("Light plot background", value=False)
    background_choice = "light" if background_toggle else "dark"

    auto_trigger = (
        st.session_state.get("has_generated_plot", False)
        and (year_changed or epsilon_changed or filters_changed or features_changed or norm_changed)
    )
    should_generate = st.button("Generate Ball Mapper graph", type="primary") or auto_trigger
    if should_generate:
        normalized, used_features = bm.normalize_features(
            filtered_df, st.session_state["bm_feature_cols"], st.session_state["bm_norm_method"]
        )
        labels = bm.compose_labels(filtered_df)
        centers, cover = bm.build_cover(normalized, epsilon)
        if not centers:
            st.warning("ε too small — no landmarks created. Increase ε.")
            return
        nodes = bm.build_nodes(centers, cover, filtered_df)
        G = bm.build_graph(nodes, labels)
        if color_metric not in nodes[0].metrics and color_metric != "size":
            st.warning(f"{color_metric} is not available as a numeric metric.")
            return
        node_df = bm.nodes_to_dataframe(nodes, labels)
        fig = build_plotly_graph(
            G,
            color_metric,
            size_metric,
            st.session_state["bm_feature_cols"],
            background=background_choice,
        )
        caption_text = f"Features: {', '.join(st.session_state['bm_feature_cols'])} | Colour: {color_metric} | ε = {epsilon}"
        caption_color = "#111111" if background_choice == "light" else "#f0f0f0"
        fig.add_annotation(
            text=caption_text,
            xref="paper",
            yref="paper",
            x=0,
            y=-0.08,
            showarrow=False,
            font=dict(color=caption_color, size=12),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption(caption_text)
        fig_html = fig.to_html(include_plotlyjs="cdn")
        png_bytes = None
        png_error = None
        try:
            png_bytes = fig.to_image(format="png", scale=2)
        except Exception as exc:  # Kaleido may be unavailable on some deployments
            png_error = str(exc)
        plot_col1, plot_col2 = st.columns(2)
        with plot_col1:
            st.download_button(
                "Download plot (HTML)",
                data=fig_html,
                file_name="health_index_ballmapper_plot.html",
                mime="text/html",
                use_container_width=True,
            )
        with plot_col2:
            if png_bytes:
                st.download_button(
                    "Download plot (PNG)",
                    data=png_bytes,
                    file_name="health_index_ballmapper_plot.png",
                    mime="image/png",
                    use_container_width=True,
                )
            else:
                st.info("PNG export unavailable in this environment (Kaleido missing). Download the HTML version instead.")
        node_summary_text = summarize_nodes(node_df, color_metric)
        st.info(node_summary_text)
        st.markdown("#### Persistence barcodes")
        landmark_points = normalized[centers] if centers else np.empty((0, normalized.shape[1]))
        barcodes = compute_barcodes(landmark_points)
        barcode_fig = render_barcodes(barcodes)
        betti_numbers = summarize_betti_numbers(barcodes)
        with st.expander("Betti numbers"):
            if betti_numbers:
                cols = st.columns(len(betti_numbers))
                for idx, (label, value) in enumerate(betti_numbers.items()):
                    cols[idx].metric(label, value)
            else:
                st.write("No Betti numbers available.")
        st.markdown("#### Geographic overlay (local authority view)")
        map_fig = None
        try:
            geojson_data, geo_codes = load_geojson(str(GEOJSON_PATH) if GEOJSON_PATH else None)
            map_df = build_area_assignment_df(
                filtered_df, nodes, color_metric, st.session_state["bm_feature_cols"]
            )
            if not map_df.empty:
                valid_codes = set(geo_codes)
                map_df = map_df[map_df["Area Code"].isin(valid_codes)]
            if map_df.empty:
                st.info("No local authority rows match the boundary file after filtering.")
            else:
                map_fig = render_geo_overlay(
                    map_df,
                    geojson_data,
                    color_metric,
                    st.session_state["bm_feature_cols"],
                )
        except FileNotFoundError as exc:
            st.info(str(exc))
            map_fig = None
        st.subheader("Node summary")
        st.dataframe(node_df)
        st.download_button(
            "Download node summary CSV",
            data=node_df.to_csv(index=False),
            file_name="health_index_ballmapper_nodes.csv",
            mime="text/csv",
        )
        st.download_button(
            "Download node summary JSON",
            data=json.dumps(node_df.to_dict(orient="records"), indent=2),
            file_name="health_index_ballmapper_nodes.json",
            mime="application/json",
        )
        st.session_state["has_generated_plot"] = True
        report_context = build_report_context(
            data_year=selected_year,
            preset_name=preset_name,
            preset_notes=preset_cfg.get("notes", ""),
            epsilon=epsilon,
            normalization=st.session_state["bm_norm_method"],
            feature_cols=st.session_state["bm_feature_cols"],
            color_metric=color_metric,
            size_metric=size_metric,
            sources=selected_sources,
            area_types=selected_area_types,
            filtered_df=filtered_df,
            node_df=node_df,
            node_summary=node_summary_text,
            ballmapper_fig=fig,
            map_fig=map_fig,
            barcode_fig=barcode_fig,
            ballmapper_png_bytes=png_bytes,
        )
        st.session_state["latest_report_context"] = report_context
        if REPORTLAB_AVAILABLE:
            try:
                pdf_bytes = render_report_pdf(report_context)
                st.download_button(
                    "Download analysis PDF",
                    data=pdf_bytes,
                    file_name="health_index_ballmapper_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            except Exception as exc:
                st.warning(f"Report generation failed: {exc}")
        else:
            st.info("Install `reportlab` to enable PDF report downloads.")


if __name__ == "__main__":
    main()
