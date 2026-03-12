"""Plotly circuit-map renderer for live and replay sessions."""

from __future__ import annotations

from typing import Any

import pandas as pd
import plotly.graph_objects as go

from src.circuit_maps import CIRCUIT_REGISTRY, circuit_coordinates, circuit_layer_geometry, map_openf1_to_svg, resolve_circuit_id


def _driver_point(row: pd.Series, circuit_id: str) -> tuple[float, float]:
    if pd.notna(row.get("svg_x")) and pd.notna(row.get("svg_y")):
        return float(row["svg_x"]), float(row["svg_y"])
    return map_openf1_to_svg(float(row.get("x", 0.0)), float(row.get("y", 0.0)), circuit_id)


def render_circuit_map(
    circuit_id: str,
    positions_df: pd.DataFrame | None = None,
    session_type: str = "race",
    title: str | None = None,
) -> go.Figure:
    """Render a circuit SVG trace with optional driver overlays."""

    resolved = resolve_circuit_id(circuit_id)
    circuit_info = CIRCUIT_REGISTRY[resolved]
    circuit_coords = circuit_coordinates(resolved)
    layers = circuit_layer_geometry(resolved)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=circuit_coords["x"],
            y=circuit_coords["y"],
            mode="lines",
            line={"color": "#4A4A57", "width": 10},
            name="Circuit",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=circuit_coords["x"],
            y=circuit_coords["y"],
            mode="lines",
            line={"color": "#E5E7EB", "width": 5},
            name="Circuit highlight",
            hoverinfo="skip",
        )
    )

    for sector_num, sector_point in layers["sector_boundaries"].items():
        fig.add_annotation(
            x=sector_point["x"],
            y=sector_point["y"],
            text=f"S{sector_num}",
            font={"color": "#FFFFFF", "size": 10, "family": "JetBrains Mono"},
            showarrow=False,
            bgcolor="rgba(14,14,20,0.75)",
            bordercolor="rgba(255,255,255,0.18)",
            borderpad=3,
        )

    for drs_zone in layers["drs_zones_coords"]:
        if not drs_zone["x"]:
            continue
        fig.add_trace(
            go.Scatter(
                x=drs_zone["x"],
                y=drs_zone["y"],
                mode="lines",
                line={"color": "#00D4A0", "width": 8, "dash": "dash"},
                name="DRS Zone",
                hoverinfo="skip",
            )
        )

    if layers["pit_lane_coords"] is not None:
        fig.add_trace(
            go.Scatter(
                x=layers["pit_lane_coords"]["x"],
                y=layers["pit_lane_coords"]["y"],
                mode="lines",
                line={"color": "#8A8A9A", "width": 4},
                name="Pit Lane",
                hoverinfo="skip",
            )
        )

    if positions_df is not None and not positions_df.empty:
        for _, driver in positions_df.iterrows():
            svg_x, svg_y = _driver_point(driver, resolved)
            fig.add_trace(
                go.Scatter(
                    x=[svg_x],
                    y=[svg_y],
                    mode="markers+text",
                    marker={
                        "size": 18,
                        "color": str(driver.get("team_color", "#888888")),
                        "line": {"color": "#FFFFFF", "width": 2},
                        "symbol": "circle",
                    },
                    text=[str(driver.get("driver_abbr", driver.get("driver_number", "?")))],
                    textposition="top center",
                    textfont={"color": "#FFFFFF", "size": 9, "family": "JetBrains Mono"},
                    name=str(driver.get("driver_abbr", driver.get("driver_number", "?"))),
                    hovertemplate=(
                        f"<b>{driver.get('driver_name', 'Unknown')}</b><br>"
                        f"P{driver.get('position_in_race', '?')}<br>"
                        f"Gap: {driver.get('gap_to_leader', '--')}<br>"
                        f"Lap: {driver.get('current_lap', '--')}<br>"
                        f"Tyre: {driver.get('compound', '--')}<br>"
                        f"Speed: {driver.get('speed', '--')} km/h"
                        "<extra></extra>"
                    ),
                )
            )
            fig.add_annotation(
                x=svg_x,
                y=svg_y,
                text=str(driver.get("position_in_race", "")),
                font={"color": "#000000", "size": 8, "family": "JetBrains Mono"},
                showarrow=False,
                yshift=-2,
                bgcolor=str(driver.get("team_color", "#FFFFFF")),
                bordercolor="#000000",
                borderwidth=0,
                opacity=0.95,
            )

    fig.update_layout(
        plot_bgcolor="#0A0A0F",
        paper_bgcolor="#0A0A0F",
        showlegend=False,
        xaxis={"showgrid": False, "zeroline": False, "showticklabels": False},
        yaxis={
            "showgrid": False,
            "zeroline": False,
            "showticklabels": False,
            "scaleanchor": "x",
            "scaleratio": 1,
        },
        margin={"l": 0, "r": 0, "t": 48, "b": 0},
        title={
            "text": title or f"{circuit_info.gp_name} - {str(session_type).title()}",
            "font": {"color": "#FFFFFF", "size": 19, "family": "Exo 2"},
            "x": 0.5,
        },
        height=500,
    )
    return fig
