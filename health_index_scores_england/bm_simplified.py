"""Minimal runner to create a Ball Mapper plot and save it as PNG + node CSV.

Edit the CONFIG section below to change the dataset, features, and styling.
Run with: `python bm_cli.py` (no CLI flags needed).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import plotly.graph_objects as go

# -----------------------
# INPUT VARIABLES HERE
# -----------------------
# Dataset and columns
CSV_PATH = Path(__file__).parent / "health_index_combined_2021.csv"
FEATURES = [
    "Healthy People Domain",
    "Healthy Lives Domain",
    "Healthy Places Domain",
]
COLOR_METRIC = "Healthy People Domain"
SIZE_METRIC = "size"  # any numeric column in the dataset, or "size" for member count

# Ball Mapper parameters
EPSILON = 0.2
BACKGROUND = "dark"  # "dark" or "light"
CSV_DELIMITER = ","

# Output files (saved alongside this script)
OUTPUT_PNG = Path(__file__).parent / "bm_plot.png"


# -----------------------
# Minimal Ball Mapper helpers (standalone, no external import)
# -----------------------

@dataclass
class BallMapperNode:
    index: int
    center_idx: int
    members: List[int]
    size: int
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


def normalize_features(df: pd.DataFrame, feature_cols: Sequence[str], method: str = "minmax") -> Tuple[np.ndarray, List[str]]:
    if not feature_cols:
        raise ValueError("Select at least one feature for the Ball Mapper point cloud.")
    numeric_df = df.copy()
    for col in feature_cols:
        numeric_df[col] = pd.to_numeric(numeric_df[col], errors="coerce")
    data = numeric_df[feature_cols].to_numpy(dtype=float)
    if method == "zscore":
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        std[std == 0] = 1.0
        normalized = (data - mean) / std
    else:
        min_vals = np.nanmin(data, axis=0)
        max_vals = np.nanmax(data, axis=0)
        denom = np.where(max_vals - min_vals == 0, 1, max_vals - min_vals)
        normalized = (data - min_vals) / denom
    normalized = np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0)
    return normalized, list(feature_cols)


def compose_labels(df: pd.DataFrame) -> List[str]:
    def _label(row: pd.Series) -> str:
        base = row.get("Area Name", "")
        type_part = row.get("Area Type", "")
        return f"{base} ({type_part})" if type_part else base

    return df.apply(_label, axis=1).tolist()


def compute_ball(points: np.ndarray, center: np.ndarray, radius: float) -> List[int]:
    distances = np.linalg.norm(points - center, axis=1)
    return sorted(np.where(distances <= radius)[0].tolist())


def build_cover(points: np.ndarray, radius: float) -> Tuple[List[int], List[List[int]]]:
    centers: List[int] = []
    cover: List[List[int]] = []
    for idx in range(points.shape[0]):
        if not centers:
            centers.append(idx)
            cover.append(compute_ball(points, points[idx], radius))
            continue
        distances = np.linalg.norm(points[centers] - points[idx], axis=1)
        if (distances <= radius).any():
            continue
        centers.append(idx)
        cover.append(compute_ball(points, points[idx], radius))
    return centers, cover


def build_nodes(
    centers: Sequence[int],
    cover: Sequence[Sequence[int]],
    df: pd.DataFrame,
) -> List[BallMapperNode]:
    numeric_cols = [
        col
        for col in df.columns
        if col not in {"Source", "Year", "Area Code", "Area Name", "Area Type"}
        and pd.api.types.is_numeric_dtype(df[col])
    ]
    nodes: List[BallMapperNode] = []
    for node_idx, (center_idx, members) in enumerate(zip(centers, cover)):
        member_array = np.asarray(members, dtype=int)
        metrics: Dict[str, float] = {}
        for col in numeric_cols:
            values = pd.to_numeric(df.iloc[member_array][col], errors="coerce")
            metrics[col] = float(np.nanmean(values)) if not values.isna().all() else 0.0
        row = df.iloc[center_idx]
        metadata = {
            "area_code": row.get("Area Code", ""),
            "area_name": row.get("Area Name", ""),
            "area_type": row.get("Area Type", ""),
            "source": row.get("Source", ""),
        }
        nodes.append(
            BallMapperNode(
                index=node_idx,
                center_idx=center_idx,
                members=list(members),
                size=len(members),
                metrics=metrics,
                metadata=metadata,
            )
        )
    return nodes


def build_graph(nodes: Sequence[BallMapperNode], labels: Sequence[str]) -> nx.Graph:
    G = nx.Graph()
    for node in nodes:
        attrs = {
            "members": node.members,
            "size": node.size,
            "label": labels[node.center_idx],
        }
        attrs.update(node.metrics)
        attrs.update(node.metadata)
        G.add_node(node.index, **attrs)
    for i in range(len(nodes)):
        members_i = set(nodes[i].members)
        for j in range(i + 1, len(nodes)):
            if members_i.intersection(nodes[j].members):
                G.add_edge(i, j)
    return G


def compute_layout(G: nx.Graph) -> dict[int, tuple[float, float]]:
    if len(G) == 0:
        return {}
    counts = [G.nodes[n]["size"] for n in G.nodes]
    avg = np.mean(counts) if counts else 1.0
    max_count = max(counts) if counts else 1.0
    base_k = 1 / np.sqrt(len(G))
    k = base_k * (1 + avg / max_count)
    try:
        return nx.spring_layout(G, seed=42, k=k, weight=None)
    except Exception:
        return nx.spectral_layout(G)


def nodes_to_dataframe(nodes: Sequence[BallMapperNode], labels: Sequence[str]) -> pd.DataFrame:
    records = []
    for node in nodes:
        record = {
            "node_id": node.index,
            "label": labels[node.center_idx],
            "size": node.size,
            "members": node.members,
        }
        record.update(node.metrics)
        record.update(node.metadata)
        records.append(record)
    return pd.DataFrame(records)


def build_plotly_graph(
    G: nx.Graph,
    color_metric: str,
    size_metric: str,
    feature_cols: List[str],
    background: str = "dark",
) -> go.Figure:
    pos = compute_layout(G)
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
        margin=dict(l=20, r=20, t=60, b=60),
        plot_bgcolor=bg_color,
        paper_bgcolor=bg_color,
        font=dict(color=font_color),
    )
    return fig


def run_ballmapper(
    df: pd.DataFrame,
    features: List[str],
    color_metric: str,
    size_metric: str,
    epsilon: float,
    background: str = "dark",
) -> tuple[go.Figure, pd.DataFrame]:
    normalized, used_features = normalize_features(df, features)
    labels = compose_labels(df)
    centers, cover = build_cover(normalized, epsilon)
    if not centers:
        raise ValueError("ε too small — no landmarks created. Increase ε.")
    nodes = build_nodes(centers, cover, df)
    G = build_graph(nodes, labels)
    node_df = nodes_to_dataframe(nodes, labels)
    if color_metric not in node_df.columns and color_metric != "size":
        raise ValueError(f"{color_metric} is not available as a numeric metric.")
    fig = build_plotly_graph(G, color_metric, size_metric, used_features, background=background)
    caption_text = f"Features: {', '.join(used_features)} | Colour: {color_metric} | ε = {epsilon}"
    fig.add_annotation(
        text=caption_text,
        xref="paper",
        yref="paper",
        x=0,
        y=-0.08,
        showarrow=False,
        font=dict(color="#f0f0f0" if background == "dark" else "#111111", size=12),
    )
    return fig, node_df


def main() -> None:
    df = pd.read_csv(CSV_PATH, delimiter=CSV_DELIMITER)
    fig, node_df = run_ballmapper(
        df,
        features=FEATURES,
        color_metric=COLOR_METRIC,
        size_metric=SIZE_METRIC,
        epsilon=EPSILON,
        background=BACKGROUND,
    )
    png_bytes = fig.to_image(format="png", scale=2)
    OUTPUT_PNG.write_bytes(png_bytes)
    print(f"Saved Ball Mapper plot to {OUTPUT_PNG}")
    nodes_out = OUTPUT_PNG.with_suffix(".nodes.csv")
    node_df.to_csv(nodes_out, index=False)
    print(f"Saved node summary to {nodes_out}")


if __name__ == "__main__":
    main()
