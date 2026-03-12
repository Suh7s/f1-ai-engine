"""Cinematic F1 AI Intelligence Platform dashboard."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dashboard.components.live_tracker import PositionSnapshot, load_replay_snapshot, load_session_snapshot, render_live_timing_tower  # noqa: E402
from dashboard.components.track_map import render_circuit_map  # noqa: E402
from src.calibration_engine import CalibrationEngine  # noqa: E402
from src.car_architecture import compute_car_circuit_fit, get_car_architecture  # noqa: E402
from src.car_performance import get_development_trajectory, predict_end_of_season_cpi  # noqa: E402
from src.championship_tracker import ChampionshipTracker, load_championship_history  # noqa: E402
from src.circuit_analysis import CIRCUIT_DEMANDS, get_circuit_specialists, load_circuit_clusters  # noqa: E402
from src.circuit_maps import CIRCUIT_REGISTRY, circuit_layer_geometry, resolve_circuit_id  # noqa: E402
from src.data_universe import openf1_position_coverage  # noqa: E402
from src.driver_database import DRIVER_DATABASE, get_driver_profile  # noqa: E402
from src.explainability import PredictionExplainer  # noqa: E402
from src.feature_engineering import build_race_inference_frame  # noqa: E402
from src.model_training import load_metrics, predict_race  # noqa: E402
from src.prediction_tracker import aggregate_season_accuracy, load_pre_race_prediction  # noqa: E402
from src.race_simulator import RaceConfig, compute_upset_probability, simulate_race  # noqa: E402
from src.registry import get_driver_name, get_season_roster  # noqa: E402
from src.season_simulator import build_remaining_races_for_season  # noqa: E402
from src.team_database import TEAM_DATABASE, get_team_profile  # noqa: E402
from src.tire_model import TireModel, compute_strategy_fit, get_circuit_strategy_profile  # noqa: E402
from src.true_skill import compare_drivers, plot_tsi_history  # noqa: E402
from src.utils import CONFIG as BASE_CONFIG, format_race_label, get_gp_name, load_dataframe, load_json, slugify  # noqa: E402
from src.weather_pipeline import fetch_race_weekend_forecast  # noqa: E402
from src.whatif_simulator import WhatIfSimulator  # noqa: E402


st.set_page_config(page_title="F1 AI Intelligence Platform", page_icon="data:image/svg+xml,%3Csvg xmlns=%27http://www.w3.org/2000/svg%27 viewBox=%270 0 64 64%27%3E%3Crect width=%2764%27 height=%2764%27 rx=%2712%27 fill=%27%230a0a0f%27/%3E%3Cpath d=%27M10 42h14l6-20h10l-3 10h17%27 stroke=%27%23e10600%27 stroke-width=%276%27 stroke-linecap=%27round%27 stroke-linejoin=%27round%27 fill=%27none%27/%3E%3C/svg%3E", layout="wide")

PROCESSED_DIR = BASE_CONFIG["processed_dir"]
RAW_DIR = BASE_CONFIG["raw_dir"]
PREDICTIONS_DIR = BASE_CONFIG["predictions_dir"]

DESIGN_SYSTEM = {
    "fonts": {
        "heading": "Exo 2",
        "mono": "JetBrains Mono",
    },
    "colors": {
        "bg_primary": "#0a0a0f",
        "bg_surface": "#13131a",
        "bg_card": "#1a1a24",
        "accent_red": "#e10600",
        "accent_gold": "#ffd700",
        "accent_green": "#00d4a0",
        "text_primary": "#ffffff",
        "text_secondary": "#8a8a9a",
    },
}


def inject_theme() -> None:
    fonts = DESIGN_SYSTEM["fonts"]
    colors = DESIGN_SYSTEM["colors"]
    font_import = f"https://fonts.googleapis.com/css2?family={fonts['heading'].replace(' ', '+')}:wght@500;600;700;800&family={fonts['mono'].replace(' ', '+')}:wght@400;600&display=swap"
    css = f"""
        <style>
        @import url('{font_import}');

        :root{{
          --bg-primary:{colors["bg_primary"]};
          --bg-surface:{colors["bg_surface"]};
          --bg-card:{colors["bg_card"]};
          --bg-hover:#22222e;
          --accent-red:{colors["accent_red"]};
          --accent-gold:{colors["accent_gold"]};
          --accent-green:{colors["accent_green"]};
          --accent-blue:#0099ff;
          --text-primary:{colors["text_primary"]};
          --text-secondary:{colors["text_secondary"]};
          --text-muted:#555566;
          --border:#2a2a3a;
        }}

        html, body, [class*="css"]{{font-family:'{fonts["mono"]}', monospace;}}
        h1, h2, h3, h4, h5, h6, .f1-title, .f1-card-value, .f1-kicker, .f1-page-title{{
          font-family:'{fonts["heading"]}', sans-serif !important;
        }}

        [data-testid="stAppViewContainer"]{{
          background:
            radial-gradient(circle at 16% 0%, rgba(225,6,0,0.22), transparent 30%),
            radial-gradient(circle at 88% 8%, rgba(0,153,255,0.10), transparent 26%),
            linear-gradient(180deg, #05060a 0%, #0a0a0f 55%, #07080d 100%);
          color:var(--text-primary);
        }}

        [data-testid="stHeader"], footer, #MainMenu{{display:none;}}
        section[data-testid="stSidebar"]{{
          background:rgba(8,8,12,0.95);
          border-right:1px solid var(--border);
        }}
        .block-container{{padding-top:1.1rem;padding-bottom:2rem;max-width:1600px;}}
        .stSelectbox label, .stSlider label, .stRadio label{{
          color:var(--text-secondary) !important;
          letter-spacing:.12em !important;
          text-transform:uppercase !important;
          font-size:.72rem !important;
          font-weight:700 !important;
        }}
        .stTabs [data-baseweb="tab-list"]{{gap:.4rem;}}
        .stTabs [data-baseweb="tab"]{{
          background:rgba(255,255,255,0.02);
          border:1px solid var(--border);
          border-radius:14px 14px 0 0;
          color:var(--text-secondary);
          text-transform:uppercase;
          letter-spacing:.06em;
        }}
        .stTabs [aria-selected="true"]{{
          background:rgba(225,6,0,0.16) !important;
          color:var(--text-primary) !important;
          border-color:rgba(225,6,0,0.38) !important;
        }}
        .stButton button, .stDownloadButton button{{
          width:100%;
          border-radius:14px;
          border:1px solid var(--border);
          background:linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.03));
          color:var(--text-primary);
          min-height:2.85rem;
          text-transform:uppercase;
          letter-spacing:.08em;
          font-weight:700;
        }}
        .stButton button[kind="primary"]{{
          background:linear-gradient(135deg, #ff2418, #8c0904);
          border-color:rgba(225,6,0,0.56);
        }}
        div[data-testid="stPlotlyChart"], div[data-testid="stDataFrame"]{{
          background:linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.015));
          border:1px solid var(--border);
          border-radius:18px;
          overflow:hidden;
        }}
        .f1-shell{{
          border:1px solid var(--border);
          border-radius:26px;
          padding:1.35rem 1.35rem 1.1rem;
          background:
            linear-gradient(135deg, rgba(225,6,0,0.18), rgba(225,6,0,0) 35%),
            linear-gradient(180deg, rgba(255,255,255,0.05), rgba(255,255,255,0.02)),
            rgba(15,15,22,0.92);
          box-shadow:0 22px 70px rgba(0,0,0,0.38);
        }}
        .f1-page-title{{font-size:3.2rem;line-height:0.92;margin:0 0 .3rem 0;}}
        .f1-kicker{{font-size:.8rem;letter-spacing:.18em;color:#ff9c96;text-transform:uppercase;font-weight:700;}}
        .f1-subtitle{{color:var(--text-secondary);font-size:.95rem;line-height:1.6;margin-top:.4rem;}}
        .f1-grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1rem;margin-top:1rem;}}
        .f1-card{{
          background:var(--bg-card);
          border:1px solid var(--border);
          border-radius:16px;
          padding:1rem 1rem .9rem;
          min-height:116px;
        }}
        .f1-card-label{{color:var(--text-secondary);font-size:.74rem;text-transform:uppercase;letter-spacing:.12em;}}
        .f1-card-value{{font-size:2rem;line-height:1;margin:.35rem 0;color:var(--text-primary);}}
        .f1-card-meta{{color:var(--text-secondary);font-size:.8rem;line-height:1.5;}}
        .f1-status{{
          display:inline-flex;align-items:center;gap:.45rem;border:1px solid var(--border);
          background:rgba(255,255,255,0.03);padding:.5rem .8rem;border-radius:999px;font-size:.8rem;
        }}
        .f1-dot{{width:10px;height:10px;border-radius:50%;}}
        .f1-list-card{{
          border:1px solid var(--border);
          border-radius:16px;
          padding:1rem;
          background:linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.01));
        }}
        .f1-note{{color:var(--text-secondary);font-size:.82rem;line-height:1.55;}}
        .f1-section{{margin-top:1.2rem;}}
        .f1-pill{{
          display:inline-flex;align-items:center;padding:.45rem .7rem;border:1px solid var(--border);
          border-radius:999px;background:rgba(255,255,255,0.03);margin-right:.45rem;font-size:.76rem;
        }}
        </style>
        """
    st.markdown(css, unsafe_allow_html=True)


def ui_card(label: str, value: str, meta: str = "") -> None:
    st.markdown(
        f"""
        <div class="f1-card">
          <div class="f1-card-label">{label}</div>
          <div class="f1-card-value">{value}</div>
          <div class="f1-card-meta">{meta}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_shell(kicker: str, title: str, subtitle: str = "") -> None:
    st.markdown(
        f"""
        <div class="f1-shell">
          <div class="f1-kicker">{kicker}</div>
          <div class="f1-page-title">{title}</div>
          <div class="f1-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def rgba(hex_color: str, alpha: float) -> str:
    value = str(hex_color or "#888888").lstrip("#")
    if len(value) == 3:
        value = "".join(ch * 2 for ch in value)
    r = int(value[0:2], 16)
    g = int(value[2:4], 16)
    b = int(value[4:6], 16)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def chart_layout(fig: go.Figure, title: str | None = None, height: int = 320) -> go.Figure:
    fig.update_layout(
        title=title,
        template="plotly_dark",
        height=height,
        paper_bgcolor=DESIGN_SYSTEM["colors"]["bg_primary"],
        plot_bgcolor=DESIGN_SYSTEM["colors"]["bg_primary"],
        font={
            "family": f"{DESIGN_SYSTEM['fonts']['mono']}, monospace",
            "color": DESIGN_SYSTEM["colors"]["text_primary"],
        },
        margin={"l": 40, "r": 20, "t": 48 if title else 24, "b": 36},
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.06)", zeroline=False)
    return fig


def normalize_tsi_score(value: Any) -> float:
    numeric = float(value) if pd.notna(value) else 0.0
    return float(1.0 / (1.0 + np.exp(numeric)))


def _load_optional(path: Path, parse_dates: list[str] | None = None) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return load_dataframe(path, parse_dates=parse_dates or [])


@st.cache_data(show_spinner=False)
def load_artifacts() -> dict[str, Any]:
    calendar = _load_optional(PROCESSED_DIR / "calendar.csv", ["date"])
    race_results = _load_optional(PROCESSED_DIR / "race_results.csv", ["race_date"])
    driver_standings = _load_optional(PROCESSED_DIR / "driver_standings.csv")
    constructor_standings = _load_optional(PROCESSED_DIR / "constructor_standings.csv")
    features = _load_optional(PROCESSED_DIR / "features.csv", ["race_date"])
    race_weather = _load_optional(PROCESSED_DIR / "race_weather.csv", ["race_date"])
    tsi = _load_optional(PROCESSED_DIR / "tsi_driver_race.csv", ["race_date"])
    circuit_tsi = _load_optional(PROCESSED_DIR / "circuit_tsi_matrix.csv", ["race_date"])
    car_architecture = _load_optional(PROCESSED_DIR / "car_architecture_summary.csv")
    coverage = _load_optional(PROCESSED_DIR / "data_universe_coverage.csv")
    if coverage.empty:
        coverage = _load_optional(PROCESSED_DIR / "openf1_weekend_coverage.csv")
    if coverage.empty:
        coverage = openf1_position_coverage()
    metrics = load_metrics() if (PROCESSED_DIR / "model_metrics.json").exists() else {}

    if not race_results.empty:
        race_results["driver_name"] = (
            race_results["driver_given_name"].fillna("").astype(str).str.strip()
            + " "
            + race_results["driver_family_name"].fillna("").astype(str).str.strip()
        ).str.strip()
    if not features.empty and "driver_name" not in features.columns:
        features["driver_name"] = features["driver_id"].map(get_driver_name)

    return {
        "calendar": calendar,
        "race_results": race_results,
        "driver_standings": driver_standings,
        "constructor_standings": constructor_standings,
        "features": features,
        "race_weather": race_weather,
        "tsi": tsi,
        "circuit_tsi": circuit_tsi,
        "car_architecture": car_architecture,
        "coverage": coverage,
        "metrics": metrics,
    }


@st.cache_resource
def get_calibration_engine() -> CalibrationEngine:
    return CalibrationEngine()


@st.cache_resource
def get_explainer() -> PredictionExplainer:
    return PredictionExplainer()


@st.cache_resource
def get_tire_model() -> TireModel:
    return TireModel()


@st.cache_resource
def get_whatif_simulator() -> WhatIfSimulator:
    return WhatIfSimulator()


def active_season(data: dict[str, Any]) -> int:
    calendar = data["calendar"]
    if calendar.empty:
        return int(datetime.now(timezone.utc).year)
    return int(calendar["season"].max())


def season_calendar(data: dict[str, Any], season: int) -> pd.DataFrame:
    return data["calendar"].loc[data["calendar"]["season"] == int(season)].sort_values("round").reset_index(drop=True)


def latest_completed_round(data: dict[str, Any], season: int) -> int:
    calendar = season_calendar(data, season)
    if calendar.empty:
        return 0
    completed = calendar.loc[calendar["is_completed"] == 1]
    if not completed.empty:
        return int(completed["round"].max())
    past = calendar.loc[calendar["date"] <= pd.Timestamp(datetime.now(timezone.utc))]
    return int(past["round"].max()) if not past.empty else 0


def active_weekend_round(data: dict[str, Any], season: int) -> int:
    calendar = season_calendar(data, season)
    if calendar.empty:
        return 0
    now = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    weekend_window = calendar.loc[
        (calendar["is_completed"] == 0)
        & (calendar["date"] >= now - pd.Timedelta(days=3))
        & (calendar["date"] <= now + pd.Timedelta(days=1))
    ]
    if weekend_window.empty:
        return 0
    return int(weekend_window.sort_values("round").iloc[0]["round"])


def next_race_row(data: dict[str, Any], season: int) -> pd.Series:
    calendar = season_calendar(data, season)
    if calendar.empty:
        return pd.Series(dtype=object)
    current_ts = pd.Timestamp(datetime.now(timezone.utc)).tz_localize(None)
    upcoming = calendar.loc[(calendar["is_completed"] == 0) & (calendar["date"] >= current_ts)]
    if upcoming.empty:
        upcoming = calendar.loc[calendar["is_completed"] == 0]
    if upcoming.empty:
        return calendar.iloc[-1]
    return upcoming.sort_values("round").iloc[0]


def last_race_row(data: dict[str, Any], season: int) -> pd.Series:
    calendar = season_calendar(data, season)
    if calendar.empty:
        return pd.Series(dtype=object)
    completed = calendar.loc[calendar["is_completed"] == 1].sort_values("round")
    if completed.empty:
        return calendar.iloc[0]
    return completed.iloc[-1]


def countdown_text(target_date: pd.Timestamp) -> str:
    target = pd.Timestamp(target_date)
    if target.tzinfo is None:
        target = target.tz_localize("UTC")
    now = pd.Timestamp(datetime.now(timezone.utc))
    delta = target - now
    if delta.total_seconds() <= 0:
        return "Live weekend"
    days = delta.days
    hours, remainder = divmod(delta.seconds, 3600)
    minutes = remainder // 60
    return f"{days} days {hours} hours {minutes} minutes"


def standings_snapshot(frame: pd.DataFrame, season: int, id_column: str) -> pd.DataFrame:
    season_frame = frame.loc[frame["season"] == int(season)].copy() if not frame.empty else pd.DataFrame()
    if season_frame.empty:
        return pd.DataFrame()
    latest_round = int(season_frame["round"].max())
    snapshot = season_frame.loc[season_frame["round"] == latest_round].sort_values("position").reset_index(drop=True)
    return snapshot[[column for column in snapshot.columns if column in ["season", "round", id_column, "position", "points", "wins", "constructor_id"]]]


def current_race_results(data: dict[str, Any], season: int, round_number: int) -> pd.DataFrame:
    results = data["race_results"]
    return results.loc[(results["season"] == int(season)) & (results["round"] == int(round_number))].copy() if not results.empty else pd.DataFrame()


def podium_for_round(data: dict[str, Any], season: int, round_number: int) -> pd.DataFrame:
    frame = current_race_results(data, season, round_number)
    if frame.empty:
        return pd.DataFrame()
    return frame.sort_values("finish_position").head(3).reset_index(drop=True)


def fallback_snapshot(data: dict[str, Any], season: int, round_number: int) -> PositionSnapshot:
    race_frame = current_race_results(data, season, round_number)
    if race_frame.empty:
        return PositionSnapshot(
            timestamp=datetime.now(timezone.utc),
            positions=pd.DataFrame(),
            track_status="NO DATA",
            safety_car_deployed=False,
            session_key=None,
            session_name="Replay",
            circuit_id="",
            messages=pd.DataFrame(),
        )
    circuit_id = resolve_circuit_id(str(race_frame["circuit_id"].iloc[0]))
    coords = render_circuit_map(circuit_id).data[0]
    x_path = np.asarray(coords["x"])
    y_path = np.asarray(coords["y"])
    ordered = race_frame.sort_values("finish_position").reset_index(drop=True)
    driver_rows: list[dict[str, Any]] = []
    for index, row in enumerate(ordered.itertuples(index=False), start=1):
        path_index = int((index / max(len(ordered), 1)) * (len(x_path) - 1)) if len(x_path) else 0
        profile = get_driver_profile(str(row.driver_id))
        team = get_team_profile(str(row.constructor_id))
        driver_rows.append(
            {
                "driver_id": row.driver_id,
                "driver_name": getattr(row, "driver_name", profile.full_name),
                "driver_abbr": profile.abbreviation,
                "team_id": row.constructor_id,
                "team_color": team.primary_color,
                "position_in_race": int(row.finish_position),
                "gap_to_leader": "Leader" if int(row.finish_position) == 1 else f"+{int(row.finish_position) - 1} pos",
                "compound": getattr(row, "primary_compound", "--"),
                "speed": np.nan,
                "current_lap": int(CIRCUIT_REGISTRY[circuit_id].laps),
                "svg_x": float(x_path[path_index]) if len(x_path) else 0.0,
                "svg_y": float(y_path[path_index]) if len(y_path) else 0.0,
            }
        )
    return PositionSnapshot(
        timestamp=datetime.now(timezone.utc),
        positions=pd.DataFrame(driver_rows),
        track_status="CHEQUERED FLAG",
        safety_car_deployed=False,
        session_key=None,
        session_name="Replay",
        circuit_id=circuit_id,
        messages=pd.DataFrame(),
    )


def replay_snapshot(data: dict[str, Any], season: int, round_number: int) -> PositionSnapshot:
    position_path = RAW_DIR / "openf1" / f"{int(season)}" / f"R{int(round_number):02d}" / "position.parquet"
    if not position_path.exists():
        return fallback_snapshot(data, season, round_number)
    snapshot = load_replay_snapshot(season, round_number, session_group="RACE")
    if snapshot.positions.empty:
        return fallback_snapshot(data, season, round_number)
    return snapshot


def prediction_inputs(data: dict[str, Any], season: int, round_number: int) -> tuple[str, list[str], dict[str, int], str]:
    historical = current_race_results(data, season, round_number)
    if not historical.empty and historical["qualifying_position"].notna().any():
        historical = historical.sort_values("qualifying_position")
        qualifying_map = {
            str(row.driver_id): int(row.qualifying_position)
            for row in historical[["driver_id", "qualifying_position"]].dropna().itertuples(index=False)
        }
        return (
            resolve_circuit_id(str(historical["circuit_id"].iloc[0])),
            historical["driver_id"].astype(str).tolist(),
            qualifying_map,
            "dry",
        )

    remaining = build_remaining_races_for_season(int(season), from_round=max(0, int(round_number) - 1))
    config = next((item for item in remaining if int(item.round) == int(round_number)), None)
    if config is not None:
        if isinstance(config.qualifying_positions, dict):
            qualifying_map = {str(key): int(value) for key, value in config.qualifying_positions.items()}
        else:
            qualifying_map = {str(driver_id): int(index + 1) for index, driver_id in enumerate(config.drivers)}
        return resolve_circuit_id(str(config.circuit_id)), list(map(str, config.drivers)), qualifying_map, str(config.weather_mode)

    roster = get_season_roster(int(season))
    circuit_id = resolve_circuit_id(str(season_calendar(data, season).loc[lambda frame: frame["round"] == int(round_number), "circuit_id"].iloc[0]))
    drivers = [str(row["driver_id"]) for row in roster]
    qualifying_map = {driver_id: index + 1 for index, driver_id in enumerate(drivers)}
    return circuit_id, drivers, qualifying_map, "dry"


@st.cache_data(show_spinner=False)
def run_prediction_bundle(season: int, round_number: int, simulations: int) -> dict[str, Any]:
    circuit_id, driver_ids, qualifying_map, weather_mode = prediction_inputs(load_artifacts(), season, round_number)
    simulation = simulate_race(
        RaceConfig(
            drivers=driver_ids,
            qualifying_positions=qualifying_map,
            circuit_id=circuit_id,
            season=season,
            round=round_number,
            weather_mode=weather_mode,
            n_simulations=int(simulations),
        )
    )
    predictions = predict_race(
        drivers_list=driver_ids,
        qualifying_positions=qualifying_map,
        circuit_id=circuit_id,
        season=season,
        round_number=round_number,
        weather_mode=weather_mode,
    )
    inference = build_race_inference_frame(
        drivers_list=driver_ids,
        qualifying_positions=qualifying_map,
        circuit_id=circuit_id,
        season=season,
        round_number=round_number,
        weather_mode=weather_mode,
    )
    confidence = get_calibration_engine().compute_prediction_confidence(inference, season=season, round_number=round_number)
    upset = compute_upset_probability(circuit_id=circuit_id, season=season, round_number=round_number, weather_forecast={"weather_mode": weather_mode})
    return {
        "circuit_id": circuit_id,
        "driver_ids": driver_ids,
        "qualifying_map": qualifying_map,
        "weather_mode": weather_mode,
        "simulation": simulation,
        "predictions": predictions,
        "inference": inference,
        "confidence": confidence,
        "upset": upset,
    }


@st.cache_data(show_spinner=False)
def title_probability_evolution(season: int, top_driver_ids: tuple[str, ...]) -> pd.DataFrame:
    tracker = ChampionshipTracker(season)
    rounds = sorted(tracker.driver_standings.loc[tracker.driver_standings["season"] == int(season), "round"].dropna().astype(int).unique().tolist())
    rows: list[dict[str, Any]] = []
    for round_number in rounds:
        probs = tracker._wdc_probability_at_round(round_number, n_simulations=1000)
        for driver_id in top_driver_ids:
            probability = 0.0
            if not probs.empty and driver_id in probs["driver_id"].tolist():
                probability = float(probs.loc[probs["driver_id"] == driver_id, "wdc_probability"].iloc[0])
            rows.append({"round": round_number, "driver_id": driver_id, "driver_name": get_driver_name(driver_id), "wdc_probability": probability})
    return pd.DataFrame(rows)


def latest_circuit_score(data: dict[str, Any], driver_id: str, circuit_id: str) -> float:
    matrix = data["circuit_tsi"]
    if matrix.empty:
        return 0.5
    rows = matrix.loc[(matrix["driver_id"] == str(driver_id)) & (matrix["circuit_id"] == str(circuit_id))].sort_values(["season", "round"])
    if rows.empty:
        return 0.5
    score = float(rows.iloc[-1].get("circuit_tsi_score_after", np.nan))
    if not np.isfinite(score):
        return 0.5
    return normalize_tsi_score(score)


def probability_bar(frame: pd.DataFrame, value_column: str, label_column: str, color_column: str, title: str, height: int = 360) -> go.Figure:
    ordered = frame.sort_values(value_column, ascending=True)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=ordered[value_column],
            y=ordered[label_column],
            orientation="h",
            marker={"color": ordered[color_column].tolist()},
            text=[f"{value:.1%}" if value <= 1.0 else f"{value:.1f}" for value in ordered[value_column]],
            textposition="outside",
        )
    )
    fig.update_xaxes(tickformat=".0%")
    return chart_layout(fig, title, height=height)


def radar_chart(values: dict[str, float], title: str, color: str = "#E10600") -> go.Figure:
    keys = list(values.keys())
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=[float(values[key]) for key in keys] + [float(values[keys[0]])],
            theta=keys + [keys[0]],
            fill="toself",
            line={"color": color},
            fillcolor=rgba(color, 0.24),
        )
    )
    fig.update_layout(
        title=title,
        height=360,
        template="plotly_dark",
        paper_bgcolor="#0a0a0f",
        plot_bgcolor="#0a0a0f",
        font={"family": "JetBrains Mono, monospace", "color": "#FFFFFF"},
        polar={"radialaxis": {"range": [0, 1], "gridcolor": "rgba(255,255,255,0.10)"}, "angularaxis": {"gridcolor": "rgba(255,255,255,0.10)"}},
        margin={"l": 30, "r": 30, "t": 48, "b": 20},
    )
    return fig


def season_timeline_figure(data: dict[str, Any], season: int) -> go.Figure:
    calendar = season_calendar(data, season)
    results = data["race_results"]
    winners = (
        results.loc[(results["season"] == int(season)) & (results["finish_position"] == 1), ["round", "driver_name", "constructor_id"]]
        if not results.empty
        else pd.DataFrame(columns=["round", "driver_name", "constructor_id"])
    )
    merged = calendar.merge(winners, how="left", on="round")
    colors = []
    labels = []
    for row in merged.itertuples(index=False):
        if int(row.is_completed) == 1:
            team = get_team_profile(str(getattr(row, "constructor_id", "")))
            colors.append(team.primary_color)
            labels.append(str(getattr(row, "driver_name", "Winner TBD")))
        else:
            colors.append("#E10600" if int(row.round) == int(next_race_row(data, season).get("round", 0)) else "#555566")
            labels.append("Upcoming")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["round"],
            y=[1] * len(merged),
            mode="markers+text",
            marker={"size": 20, "color": colors, "line": {"color": "#FFFFFF", "width": 1.5}},
            text=merged["race_name"],
            textposition="top center",
            customdata=np.stack([labels], axis=-1) if len(labels) else None,
            hovertemplate="<b>%{text}</b><br>%{customdata[0]}<extra></extra>",
        )
    )
    fig.update_yaxes(visible=False)
    fig.update_xaxes(dtick=1, title="Grand Prix calendar")
    return chart_layout(fig, "Season Overview", height=220)


def strategy_timeline(season: int, round_number: int) -> go.Figure | None:
    path = RAW_DIR / "openf1" / f"{int(season)}" / f"R{int(round_number):02d}" / "stints.parquet"
    if not path.exists():
        return None
    frame = load_dataframe(path)
    if frame.empty:
        return None
    color_map = {"soft": "#E10600", "medium": "#FFD700", "hard": "#FFFFFF", "intermediate": "#00D4A0", "wet": "#0099FF"}
    fig = go.Figure()
    ordered_drivers = sorted(frame["driver_number"].astype(str).unique().tolist())
    for driver_number in ordered_drivers:
        driver_frame = frame.loc[frame["driver_number"].astype(str) == str(driver_number)].sort_values("lap_start")
        label = get_driver_profile(driver_frame["driver_id"].iloc[0]).abbreviation if "driver_id" in driver_frame.columns and driver_frame["driver_id"].notna().any() else str(driver_number)
        for stint in driver_frame.itertuples(index=False):
            start_lap = float(getattr(stint, "lap_start", 0) or 0)
            end_lap = float(getattr(stint, "lap_end", start_lap) or start_lap)
            compound = str(getattr(stint, "compound", "hard")).lower()
            fig.add_trace(
                go.Bar(
                    x=[max(end_lap - start_lap + 1, 1)],
                    y=[label],
                    base=[start_lap],
                    orientation="h",
                    marker={"color": color_map.get(compound, "#888888")},
                    hovertemplate=f"{label} - {compound.title()}<br>Laps {int(start_lap)}-{int(end_lap)}<extra></extra>",
                    showlegend=False,
                )
            )
    return chart_layout(fig, "Tyre Strategy Tracker", height=380)


def gap_figure(season: int, round_number: int) -> go.Figure | None:
    path = RAW_DIR / "openf1" / f"{int(season)}" / f"R{int(round_number):02d}" / "intervals.parquet"
    if not path.exists():
        return None
    frame = load_dataframe(path)
    if frame.empty or "date" not in frame.columns:
        return None
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).copy()
    if frame.empty:
        return None
    latest_drivers = frame["driver_number"].astype(str).dropna().unique().tolist()[:8]
    fig = go.Figure()
    for driver_number in latest_drivers:
        driver_frame = frame.loc[frame["driver_number"].astype(str) == str(driver_number)].sort_values("date")
        profile = get_driver_profile(driver_frame["driver_id"].iloc[0]) if "driver_id" in driver_frame.columns and driver_frame["driver_id"].notna().any() else get_driver_profile(str(driver_number))
        team = get_team_profile(profile.current_team or "")
        fig.add_trace(
            go.Scatter(
                x=driver_frame["date"],
                y=pd.to_numeric(driver_frame.get("gap_to_leader"), errors="coerce"),
                mode="lines",
                name=profile.abbreviation,
                line={"color": team.primary_color, "width": 2},
            )
        )
    return chart_layout(fig, "Gap To Leader", height=380)


def race_control_table(snapshot: PositionSnapshot) -> None:
    messages = snapshot.messages.copy()
    if messages.empty:
        st.info("No race control feed has been cached for this session.")
        return
    display_columns = [column for column in ["date", "flag", "category", "message", "driver_number"] if column in messages.columns]
    st.dataframe(messages[display_columns], use_container_width=True, height=380)


def render_command_center(data: dict[str, Any], season: int) -> None:
    next_race = next_race_row(data, season)
    last_race = last_race_row(data, season)
    forecast = fetch_race_weekend_forecast(season, int(next_race["round"])) if not next_race.empty else None
    accuracy = aggregate_season_accuracy(season)
    podium = podium_for_round(data, season, int(last_race["round"])) if not last_race.empty else pd.DataFrame()

    ui_shell(
        "Command Center",
        f"{str(next_race.get('race_name', 'Next race'))}",
        f"{str(next_race.get('circuit_name', ''))} - countdown {countdown_text(pd.Timestamp(next_race['date'])) if 'date' in next_race else 'n/a'}",
    )

    left, right = st.columns([1.5, 1.0], gap="large")
    with left:
        st.plotly_chart(render_circuit_map(str(next_race.get("circuit_id", "bahrain")), session_type="hero", title=str(next_race.get("race_name", "Next race"))), use_container_width=True)
    with right:
        cols = st.columns(2)
        with cols[0]:
            ui_card("Race Weekend", countdown_text(pd.Timestamp(next_race["date"])), f"{next_race.get('country', '')}")
        with cols[1]:
            weather_text = "Forecast pending"
            if forecast is not None:
                weather_text = f"{forecast.race_day_temperature:.1f} C | {forecast.race_day_rain_probability:.0f}% rain"
            ui_card("Race Day Weather", weather_text, f"Confidence {forecast.confidence if forecast else 'n/a'}")
        cols = st.columns(2)
        with cols[0]:
            ui_card("Last Race", str(last_race.get("race_name", "n/a")), str(last_race.get("circuit_name", last_race.get("country", ""))))
        with cols[1]:
            podium_accuracy = accuracy.get("per_race", [])[-1]["podium_correct"] if accuracy.get("per_race") else 0
            ui_card("Model Podium Hit", f"{podium_accuracy}/3", "Most recent scored race")

        if forecast is not None:
            pill_markup = "".join(
                [
                    f"<span class='f1-pill'>{session} {float(payload['temperature_c']):.1f}C</span>"
                    if payload.get("temperature_c") is not None
                    else f"<span class='f1-pill'>{session} forecast pending</span>"
                    for session, payload in forecast.session_forecasts.items()
                ]
            )
            st.markdown(pill_markup, unsafe_allow_html=True)

    if not podium.empty:
        st.markdown("#### Last Race Podium")
        cols = st.columns(3)
        for column, row in zip(cols, podium.itertuples(index=False), strict=False):
            team = get_team_profile(str(row.constructor_id))
            with column:
                ui_card(
                    f"P{int(row.finish_position)}",
                    str(row.driver_name),
                    f"{team.short_name} | {float(row.points):.0f} pts",
                )

    st.markdown("#### Championship Battle")
    driver_snapshot = standings_snapshot(data["driver_standings"], season, "driver_id")
    constructor_snapshot = standings_snapshot(data["constructor_standings"], season, "constructor_id")
    tracker = ChampionshipTracker(season)
    probs = tracker.get_wdc_probability(n_simulations=3000)
    if not driver_snapshot.empty:
        driver_snapshot = driver_snapshot.merge(probs, how="left", on="driver_id").fillna({"wdc_probability": 0.0})
        driver_snapshot["driver_name"] = driver_snapshot["driver_id"].map(get_driver_name)
        driver_snapshot["team_color"] = driver_snapshot["constructor_id"].map(lambda team_id: get_team_profile(str(team_id)).primary_color)
    if not constructor_snapshot.empty:
        constructor_snapshot["constructor_name"] = constructor_snapshot["constructor_id"].map(lambda team_id: get_team_profile(str(team_id)).short_name)
        constructor_snapshot["team_color"] = constructor_snapshot["constructor_id"].map(lambda team_id: get_team_profile(str(team_id)).primary_color)

    left, right = st.columns(2, gap="large")
    with left:
        if not driver_snapshot.empty:
            st.plotly_chart(probability_bar(driver_snapshot.head(3), "wdc_probability", "driver_name", "team_color", "WDC Top 3", height=300), use_container_width=True)
    with right:
        if not constructor_snapshot.empty:
            max_points = float(constructor_snapshot["points"].max() or 1.0)
            constructor_snapshot["point_share"] = constructor_snapshot["points"] / max_points
            st.plotly_chart(probability_bar(constructor_snapshot.head(3), "point_share", "constructor_name", "team_color", "WCC Top 3", height=300), use_container_width=True)

    st.plotly_chart(season_timeline_figure(data, season), use_container_width=True)


def render_live_race(data: dict[str, Any], season: int) -> None:
    target_round = active_weekend_round(data, season)
    prefer_live = target_round > 0
    if target_round <= 0:
        target_round = latest_completed_round(data, season)
    if target_round <= 0:
        target_round = int(next_race_row(data, season).get("round", 1))

    snapshot_state_key = f"live_snapshot_{season}_{target_round}"
    poll_state_key = f"live_snapshot_poll_{season}_{target_round}"
    previous_snapshot = st.session_state.get(snapshot_state_key) if prefer_live else None
    last_poll_time = st.session_state.get(poll_state_key) if prefer_live else None
    snapshot = load_session_snapshot(
        season,
        target_round,
        session_group="RACE",
        prefer_live=prefer_live,
        last_poll_time=last_poll_time,
        previous_snapshot=previous_snapshot,
    )
    if prefer_live:
        st.session_state[snapshot_state_key] = snapshot
        st.session_state[poll_state_key] = snapshot.timestamp
    race_label = format_race_label(season, target_round)
    subtitle = "Live telemetry center." if prefer_live and snapshot.session_key is not None else f"Replay center for {snapshot.session_name.lower()} with live-track-map layout."
    ui_shell("Live Race", race_label, subtitle)

    current_lap = int(pd.to_numeric(snapshot.positions.get("current_lap"), errors="coerce").fillna(0).max()) if not snapshot.positions.empty and "current_lap" in snapshot.positions.columns else 0
    total_laps = CIRCUIT_REGISTRY.get(resolve_circuit_id(snapshot.circuit_id or str(current_race_results(data, season, target_round)["circuit_id"].iloc[0] if not current_race_results(data, season, target_round).empty else "bahrain"))).laps if snapshot.circuit_id or not current_race_results(data, season, target_round).empty else 0
    dot_color = {"GREEN": "#00D4A0", "SAFETY CAR": "#FFD700", "VIRTUAL SAFETY CAR": "#FFD700", "RED FLAG": "#E10600", "CHEQUERED FLAG": "#FFFFFF"}.get(snapshot.track_status, "#8A8A9A")
    st.markdown(
        f"""
        <div class="f1-list-card">
          <div class="f1-status"><span class="f1-dot" style="background:{dot_color};"></span>{snapshot.track_status}</div>
          <span class="f1-pill">Lap {current_lap or total_laps}/{total_laps or '--'}</span>
          <span class="f1-pill">{snapshot.session_name}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if prefer_live and snapshot.session_key is not None:
        st.markdown(
            f"<div class='f1-note'>Live polling is enabled for session {snapshot.session_key}. Reloading the page will continue from the previous snapshot instead of refetching the full stream.</div>",
            unsafe_allow_html=True,
        )

    left, right = st.columns([1.6, 1.0], gap="large")
    with left:
        if snapshot.circuit_id:
            st.plotly_chart(render_circuit_map(snapshot.circuit_id, snapshot.positions, session_type=snapshot.session_name), use_container_width=True)
        else:
            st.info("No cached track map is available for this session yet.")
    with right:
        render_live_timing_tower(snapshot)

    col1, col2, col3 = st.columns(3, gap="large")
    with col1:
        gap_fig = gap_figure(season, target_round)
        if gap_fig is not None:
            st.plotly_chart(gap_fig, use_container_width=True)
        else:
            st.info("Gap history becomes available once interval data has been cached.")
    with col2:
        strategy_fig = strategy_timeline(season, target_round)
        if strategy_fig is not None:
            st.plotly_chart(strategy_fig, use_container_width=True)
        else:
            st.info("Tyre strategy data becomes available once OpenF1 stint data is ingested.")
    with col3:
        race_control_table(snapshot)


def render_race_predictor(data: dict[str, Any], season: int) -> None:
    calendar = season_calendar(data, season)
    options = {format_race_label(season, int(row.round)): int(row.round) for row in calendar.itertuples(index=False)}
    selected_label = st.selectbox("Grand Prix", list(options.keys()), index=min(max(latest_completed_round(data, season), 0), max(len(options) - 1, 0)))
    simulations = st.slider("Monte Carlo simulations", 2000, 20000, 6000, 1000)
    round_number = options[selected_label]
    bundle = run_prediction_bundle(season, round_number, simulations)
    circuit_id = bundle["circuit_id"]
    info = CIRCUIT_REGISTRY[resolve_circuit_id(circuit_id)]
    forecast = fetch_race_weekend_forecast(season, round_number)
    ui_shell("Race Predictor", info.gp_name, f"{info.full_name} | lap length {info.lap_length_km:.3f} km | {info.laps} laps | {info.drs_zones} DRS zones")

    top, side = st.columns([1.3, 1.0], gap="large")
    with top:
        st.plotly_chart(render_circuit_map(circuit_id, session_type="prediction", title=info.gp_name), use_container_width=True)
    with side:
        cols = st.columns(2)
        with cols[0]:
            ui_card("Lap Record", f"{info.lap_record.time}", f"{info.lap_record.driver} ({info.lap_record.year})")
        with cols[1]:
            ui_card("Upset Score", f"{bundle['upset']:.0%}", f"Weather mode {bundle['weather_mode']}")
        cols = st.columns(2)
        with cols[0]:
            ui_card("Confidence", bundle["confidence"].confidence_label.title(), f"Score {bundle['confidence'].confidence_score:.2f}")
        with cols[1]:
            weather_summary = "No forecast"
            if forecast is not None:
                weather_summary = f"{forecast.race_day_temperature:.1f} C | {forecast.race_day_rain_probability:.0f}% rain"
            ui_card("Race Weather", weather_summary, f"{forecast.confidence if forecast else 'historical'} confidence")

    probabilities = bundle["simulation"].win_probs.copy().sort_values("relative_win_probability", ascending=False).head(12)
    probabilities["team_color"] = probabilities["constructor_id"].map(lambda team_id: get_team_profile(str(team_id)).primary_color)
    st.plotly_chart(probability_bar(probabilities, "relative_win_probability", "driver_name", "team_color", "Win Probability"), use_container_width=True)

    advantage_rows: list[dict[str, Any]] = []
    for row in probabilities.itertuples(index=False):
        car_fit = compute_car_circuit_fit(str(row.constructor_id), circuit_id, season, round_number)
        strategy_fit = compute_strategy_fit(str(row.constructor_id), circuit_id, season, round_number)
        affinity = latest_circuit_score(data, str(row.driver_id), circuit_id)
        advantage_rows.append(
            {
                "driver_name": row.driver_name,
                "driver_affinity": affinity,
                "car_fit": float(car_fit.fit_score),
                "strategy_fit": float(strategy_fit.score),
                "team_color": get_team_profile(str(row.constructor_id)).primary_color,
            }
        )
    advantage = pd.DataFrame(advantage_rows)
    if not advantage.empty:
        fig = go.Figure()
        for column in ["driver_affinity", "car_fit", "strategy_fit"]:
            fig.add_trace(go.Bar(x=advantage["driver_name"], y=advantage[column], name=column.replace("_", " ").title()))
        st.plotly_chart(chart_layout(fig, "Circuit Advantage Breakdown", height=360), use_container_width=True)

    pred_cols = [col for col in ["driver_id", "constructor_id", "constructor_name", "expected_finish"] if col in bundle["predictions"].columns]
    expected = bundle["simulation"].expected_points.merge(
        bundle["predictions"][pred_cols],
        how="left",
        on="driver_id",
    ).sort_values("expected_finish" if "expected_finish" in pred_cols else "expected_points", ascending=False if "expected_finish" not in pred_cols else True).reset_index(drop=True)
    
    expected["driver_name"] = expected["driver_id"].map(get_driver_name)
    
    if "constructor_name" not in expected.columns and "constructor_id" in expected.columns:
        expected["constructor_name"] = expected["constructor_id"].map(lambda t: get_team_profile(str(t)).short_name if pd.notna(t) else "Unknown")
    elif "constructor_name_y" in expected.columns:
        expected["constructor_name"] = expected["constructor_name_y"]
    
    if "constructor_id" not in expected.columns:
        expected["constructor_id"] = "unknown"
        
    display_cols = [c for c in ["driver_name", "constructor_name", "expected_finish", "expected_points", "expected_points_p10", "expected_points_p90"] if c in expected.columns]
    st.dataframe(expected[display_cols], use_container_width=True, height=440)

    strategies = []
    for row in expected.head(10).itertuples(index=False):
        recommendation = get_tire_model().predict_optimal_strategy(
            driver_id=str(row.driver_id),
            circuit_id=circuit_id,
            grid_position=int(bundle["qualifying_map"].get(str(row.driver_id), 20)),
            competitors=list(map(str, expected["driver_id"].tolist())),
            weather_forecast={"weather_mode": bundle["weather_mode"]},
        )
        strategies.append(
            {
                "driver_name": get_driver_name(str(row.driver_id)),
                "strategy": recommendation.recommended_strategy,
                "alternative": recommendation.alternative_strategy,
                "undercut_lap": recommendation.undercut_lap,
                "confidence": recommendation.strategy_confidence,
            }
        )
    st.dataframe(pd.DataFrame(strategies), use_container_width=True, height=300)

    st.markdown("#### Data Universe Integration")
    st.markdown("<div class='f1-note' style='margin-bottom:1rem;'>Raw features fed directly into the XGBoost engine for this specific prediction simulation, pulled dynamically from the OpenF1 `/car_data` and `/weather` telemetry streams alongside static circuit physics profiles. Showing top 3 simulated leaders:</div>", unsafe_allow_html=True)
    
    universe_cols = st.columns(3)
    for col, row in zip(universe_cols, expected.head(3).itertuples(index=False), strict=False):
        driver_inference = bundle["inference"].loc[bundle["inference"]["driver_id"] == str(row.driver_id)]
        if not driver_inference.empty:
            speed_trap = driver_inference.get("max_speed_trap", pd.Series([np.nan])).iloc[0]
            improve_ms = driver_inference.get("fp_improvement_fp1_to_fp3", pd.Series([np.nan])).iloc[0]
            weather_tmp = driver_inference.get("race_air_temp_c", pd.Series([np.nan])).iloc[0]
            tyre_deg = driver_inference.get("circuit_tyre_deg_severity", pd.Series([np.nan])).iloc[0]
            track_risk = driver_inference.get("track_risk", pd.Series([np.nan])).iloc[0]
            
            with col:
                st.markdown(f"<div class='f1-list-card' style='border-top:3px solid {get_team_profile(str(row.constructor_id)).primary_color};'>", unsafe_allow_html=True)
                st.markdown(f"**{get_driver_name(str(row.driver_id))}**")
                st.markdown(f"• **Speed Trap**: `{speed_trap:.0f} km/h`" if pd.notna(speed_trap) else "• **Speed Trap**: `pending`")
                st.markdown(f"• **Setup Delta (FP1→FP3)**: `{-improve_ms / 1000.0:.2f}s`" if pd.notna(improve_ms) else "• **Setup Delta**: `pending`")
                st.markdown(f"• **Weather Surface T°**: `{weather_tmp:.1f}°C`" if pd.notna(weather_tmp) else "• **Weather Surface T°**: `simulated default`")
                st.markdown(f"• **Tyre Deg Profile**: `Level {tyre_deg:.0f}`" if pd.notna(tyre_deg) else "• **Tyre Deg Profile**: `n/a`")
                st.markdown(f"• **Track DNF Risk**: `{(track_risk * 100):.1f}%`" if pd.notna(track_risk) else "• **Track DNF Risk**: `n/a`")
                st.markdown("</div>", unsafe_allow_html=True)
                
    st.markdown("<br/>", unsafe_allow_html=True)

    explainer = get_explainer()
    driver_choice = st.selectbox("Explain AI decision factors for driver", expected["driver_name"].tolist(), key="predictor_explain")
    driver_id = expected.loc[expected["driver_name"] == driver_choice, "driver_id"].iloc[0]
    explanation = explainer.explain_race_prediction(str(driver_id), bundle["inference"], "E")
    reason_html = "".join([f"<div class='f1-pill'>{reason}</div>" for reason in explanation.top_reasons])
    st.markdown(
        f"""
        <div class="f1-list-card">
          <div class="f1-kicker">Deep SHAP Explainer</div>
          <h3 style="margin:.2rem 0 .7rem 0;font-family:'Exo 2',sans-serif;">{driver_choice}</h3>
          <div class="f1-note">Prediction model confidence: {explanation.prediction_value:.3f}. Ingesting {bundle["inference"].shape[1]} unique telemetry and parameter dimensions. Top variables pulling the engine prediction in this direction:</div>
          <div style="margin-top:.8rem;">{reason_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_championship_intelligence(data: dict[str, Any], season: int) -> None:
    tracker = ChampionshipTracker(season)
    standings = tracker._season_driver_standings()
    probabilities = tracker.get_wdc_probability(n_simulations=5000)
    standings = standings.merge(probabilities, how="left", on="driver_id").fillna({"wdc_probability": 0.0})
    standings["driver_name"] = standings["driver_id"].map(get_driver_name)
    standings["team_color"] = standings["constructor_id"].map(lambda team_id: get_team_profile(str(team_id)).primary_color)
    ui_shell("Championship Intelligence", f"{season} Title Fight", "Probability evolution, current scenarios, and championship control room.")

    if standings.empty:
        st.info("Championship data is not available yet.")
        return

    top_three = standings.sort_values("wdc_probability", ascending=False).head(3).reset_index(drop=True)
    cols = st.columns(3)
    for column, row in zip(cols, top_three.itertuples(index=False), strict=False):
        with column:
            ui_card(row.driver_name, f"{float(row.wdc_probability):.0%}", f"{float(row.points):.0f} points | {get_team_profile(str(row.constructor_id)).short_name}")

    season_points = data["race_results"].loc[data["race_results"]["season"] == int(season)].copy()
    if not season_points.empty:
        winners = season_points.loc[season_points["finish_position"] == 1, ["round", "driver_name", "constructor_id"]]
        fig = go.Figure()
        for driver_id in top_three["driver_id"].tolist():
            driver_rows = season_points.loc[season_points["driver_id"] == str(driver_id)].sort_values("round").copy()
            driver_rows["cumulative_points"] = pd.to_numeric(driver_rows["points"], errors="coerce").cumsum()
            fig.add_trace(
                go.Scatter(
                    x=driver_rows["round"],
                    y=driver_rows["cumulative_points"],
                    mode="lines+markers",
                    name=get_driver_name(str(driver_id)),
                    line={"color": get_team_profile(get_driver_profile(str(driver_id)).current_team or "").primary_color, "width": 3},
                )
            )
        st.plotly_chart(chart_layout(fig, "Season Timeline", height=360), use_container_width=True)

    evolution = title_probability_evolution(season, tuple(top_three["driver_id"].tolist()))
    if not evolution.empty:
        fig = go.Figure()
        for driver_id, driver_frame in evolution.groupby("driver_id", sort=False):
            fig.add_trace(
                go.Scatter(
                    x=driver_frame["round"],
                    y=driver_frame["wdc_probability"],
                    mode="lines+markers",
                    name=str(driver_frame["driver_name"].iloc[0]),
                    line={"width": 3, "color": get_team_profile(get_driver_profile(str(driver_id)).current_team or "").primary_color},
                )
            )
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(chart_layout(fig, "WDC Probability Evolution", height=360), use_container_width=True)

    st.markdown("#### Championship Scenarios")
    cols = st.columns(min(3, len(top_three)))
    for column, row in zip(cols, top_three.itertuples(index=False), strict=False):
        scenario = tracker.get_championship_scenarios(str(row.driver_id))
        with column:
            ui_card(
                f"What {row.driver_name} needs",
                f"{float(row.wdc_probability):.0%}",
                scenario.required_results,
            )

    history = load_championship_history()
    if not history.empty:
        st.dataframe(history.tail(15).sort_values("season", ascending=False), use_container_width=True, height=320)


def render_driver_intelligence(data: dict[str, Any], season: int) -> None:
    driver_options = sorted([(profile.full_name, driver_id) for driver_id, profile in DRIVER_DATABASE.items()])
    selected_name, driver_id = st.selectbox("Driver", driver_options, format_func=lambda item: item[0])
    profile = get_driver_profile(driver_id)
    ui_shell("Driver Intelligence", profile.full_name, f"{profile.nationality} | car #{profile.number or '--'} | {get_team_profile(profile.current_team or '').short_name}")

    cols = st.columns(4)
    with cols[0]:
        ui_card("Races", str(profile.career_stats.races), f"Debut {profile.debut_race}")
    with cols[1]:
        ui_card("Wins", str(profile.career_stats.wins), f"Podiums {profile.career_stats.podiums}")
    with cols[2]:
        ui_card("Poles", str(profile.career_stats.poles), f"Fastest laps {profile.career_stats.fastest_laps}")
    with cols[3]:
        ui_card("Championships", str(profile.career_stats.championships), ", ".join(map(str, profile.championships)) if profile.championships else "No title")

    st.plotly_chart(plot_tsi_history([driver_id]), use_container_width=True)

    circuit_matrix = data["circuit_tsi"].loc[data["circuit_tsi"]["driver_id"] == str(driver_id)].copy() if not data["circuit_tsi"].empty else pd.DataFrame()
    if not circuit_matrix.empty:
        latest = circuit_matrix.sort_values(["season", "round"]).groupby("circuit_id", as_index=False).tail(1)
        latest["normalized_circuit_score"] = latest["circuit_tsi_score_after"].map(normalize_tsi_score)
        heat = go.Figure(
            data=go.Heatmap(
                z=[latest["normalized_circuit_score"].tolist()],
                x=latest["circuit_id"].tolist(),
                y=[profile.abbreviation],
                colorscale=[[0.0, "#5A0D0D"], [0.5, "#D1A000"], [1.0, "#00D4A0"]],
            )
        )
        st.plotly_chart(chart_layout(heat, "Circuit Heatmap", height=240), use_container_width=True)

        archetypes = {
            "Power": ["monza", "spa", "vegas", "baku"],
            "Technical": ["monaco", "hungaroring", "singapore", "imola"],
            "Street": ["monaco", "baku", "jeddah", "marina_bay", "vegas", "miami"],
            "High Speed": ["silverstone", "suzuka", "spa", "zandvoort"],
            "High Deg": ["bahrain", "catalunya", "hungaroring"],
        }
        radar_values = {}
        for label, circuits in archetypes.items():
            subset = latest.loc[latest["circuit_id"].isin([resolve_circuit_id(item) for item in circuits])]
            radar_values[label] = float(subset["normalized_circuit_score"].mean()) if not subset.empty else 0.5
        st.plotly_chart(radar_chart(radar_values, "Circuit Archetype Performance", color=profile.car_color), use_container_width=True)

    comparison_target = st.selectbox("Head to head", [name for name, _ in driver_options if name != selected_name], key="driver_h2h")
    opponent_id = dict(driver_options)[comparison_target]
    comparison = compare_drivers(driver_id, opponent_id)
    left, right = st.columns(2)
    with left:
        ui_card("Driver H2H", f"{comparison['direct_head_to_head']['driver_1_ahead_pct']:.0%}" if pd.notna(comparison["direct_head_to_head"]["driver_1_ahead_pct"]) else "n/a", f"{selected_name} ahead when direct data exists")
    with right:
        ui_card("Qualifying Delta", f"{comparison['direct_head_to_head']['average_qualifying_delta_pct']:.3f}" if pd.notna(comparison["direct_head_to_head"]["average_qualifying_delta_pct"]) else "n/a", comparison_target)


def render_team_intelligence(data: dict[str, Any], season: int) -> None:
    team_options = sorted([(profile.short_name, team_id) for team_id, profile in TEAM_DATABASE.items()])
    selected_name, team_id = st.selectbox("Team", team_options, format_func=lambda item: item[0], key="team_pick")
    profile = get_team_profile(team_id)
    latest_round = max(latest_completed_round(data, season), 1)
    architecture = get_car_architecture(team_id, season, latest_round)
    ui_shell("Team Intelligence", profile.full_name, f"{profile.base} | Team principal {profile.team_principal}")

    cols = st.columns(4)
    with cols[0]:
        ui_card("Founded", str(profile.founded), profile.nationality)
    with cols[1]:
        ui_card("Titles", str(len(profile.championships)), ", ".join(map(str, profile.championships[-4:])) if profile.championships else "No WCC titles")
    with cols[2]:
        ui_card("Power Unit", profile.power_unit_supplier, "Current supplier")
    with cols[3]:
        ui_card("Current Drivers", str(len(profile.current_drivers)), ", ".join(get_driver_name(driver_id) for driver_id in profile.current_drivers) if profile.current_drivers else "No active roster")

    radar_values = {
        "Straight": architecture.straight_line_speed,
        "Slow Corners": architecture.slow_corner_grip,
        "High Speed": architecture.high_speed_cornering,
        "Mechanical": architecture.mechanical_grip,
        "Tyre Mgmt": architecture.tyre_management,
        "Downforce": architecture.downforce_level,
    }
    st.plotly_chart(radar_chart(radar_values, "Car Architecture Radar", color=profile.primary_color), use_container_width=True)

    trajectory = get_development_trajectory(team_id, season)
    if not trajectory.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=trajectory["round"], y=trajectory["cpi_performance"], mode="lines+markers", line={"color": profile.primary_color, "width": 3}, name="CPI"))
        if "cpi_rolling_3" in trajectory.columns:
            fig.add_trace(go.Scatter(x=trajectory["round"], y=trajectory["cpi_rolling_3"], mode="lines", line={"color": profile.secondary_color, "dash": "dash"}, name="Rolling 3"))
        st.plotly_chart(chart_layout(fig, "Development Trajectory", height=340), use_container_width=True)

    fit_rows = []
    calendar = season_calendar(data, season)
    for circuit_id in calendar["circuit_id"].dropna().astype(str).unique().tolist():
        car_fit = compute_car_circuit_fit(team_id, circuit_id, season, latest_round)
        fit_rows.append({"circuit_id": circuit_id, "fit_score": float(car_fit.fit_score)})
    fit_frame = pd.DataFrame(fit_rows).sort_values("fit_score", ascending=False)
    if not fit_frame.empty:
        fig = px.imshow([fit_frame["fit_score"].tolist()], x=fit_frame["circuit_id"].tolist(), y=[profile.short_name], color_continuous_scale=["#6C1010", "#D1A000", "#00D4A0"])
        st.plotly_chart(chart_layout(fig, "Circuit Fit Matrix", height=240), use_container_width=True)

    projection = predict_end_of_season_cpi(team_id, season)
    st.markdown(f"<div class='f1-note'>Projected end-of-season CPI {projection.get('projected_cpi', np.nan):.3f} with interval {projection.get('confidence_interval', ('n/a', 'n/a'))}.</div>", unsafe_allow_html=True)


def render_circuit_intelligence(data: dict[str, Any], season: int) -> None:
    circuit_options = sorted([(info.gp_name, circuit_id) for circuit_id, info in CIRCUIT_REGISTRY.items() if circuit_id == info.circuit_id])
    selected_name, circuit_id = st.selectbox("Circuit", circuit_options, format_func=lambda item: item[0], key="circuit_pick")
    info = CIRCUIT_REGISTRY[circuit_id]
    ui_shell("Circuit Intelligence", info.gp_name, f"{info.full_name} | {info.country} | first GP {info.first_gp}")

    left, right = st.columns([1.3, 1.0], gap="large")
    with left:
        st.plotly_chart(render_circuit_map(circuit_id, session_type="profile", title=info.gp_name), use_container_width=True)
    with right:
        cols = st.columns(2)
        with cols[0]:
            ui_card("Lap Length", f"{info.lap_length_km:.3f} km", f"{info.laps} laps")
        with cols[1]:
            ui_card("Lap Record", info.lap_record.time, f"{info.lap_record.driver} ({info.lap_record.year})")
        layers = circuit_layer_geometry(circuit_id)
        st.markdown(f"<div class='f1-note'>Sector markers loaded: {', '.join(f'S{key}' for key in layers['sector_boundaries'])}. DRS zones mapped: {info.drs_zones}.</div>", unsafe_allow_html=True)

    demand = CIRCUIT_DEMANDS[circuit_id]
    demand_values = {
        "Straight": float(demand.straight_line_speed),
        "Slow": float(demand.slow_corner_grip),
        "High Speed": float(demand.high_speed_cornering),
        "Mechanical": float(demand.mechanical_grip),
        "Tyre": float(demand.tyre_management),
    }
    st.plotly_chart(radar_chart(demand_values, "Circuit DNA", color="#00D4A0"), use_container_width=True)

    strategy = get_circuit_strategy_profile(circuit_id)
    st.markdown(
        f"""
        <div class="f1-list-card">
          <div class="f1-kicker">Strategy Profile</div>
          <div class="f1-note">Typical stops {strategy.typical_stops}. Pit window {strategy.pit_window_open}-{strategy.pit_window_close}. Undercut power {strategy.undercut_power:.2f}. Safety car probability {strategy.safety_car_probability:.2f}.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    specialists = get_circuit_specialists(circuit_id, season=season, round_number=max(latest_completed_round(data, season), 1), top_n=10)
    if not specialists.empty:
        specialists["driver_name"] = specialists["driver_id"].map(get_driver_name)
        specialists["specialist_score"] = specialists["circuit_tsi_score_after"].map(normalize_tsi_score)
        specialists["team_color"] = specialists["driver_id"].map(lambda driver_id: get_team_profile(get_driver_profile(str(driver_id)).current_team or "").primary_color)
        st.plotly_chart(probability_bar(specialists, "specialist_score", "driver_name", "team_color", "All-Time Specialists", height=380), use_container_width=True)

    weather = data["race_weather"].loc[data["race_weather"]["circuit_id"] == circuit_id].copy() if not data["race_weather"].empty else pd.DataFrame()
    if not weather.empty:
        monthly = weather.copy()
        monthly["month"] = pd.to_datetime(monthly["race_date"], errors="coerce").dt.month
        history = monthly.groupby("month", as_index=False)["race_precipitation_mm"].mean()
        fig = go.Figure(go.Bar(x=history["month"], y=history["race_precipitation_mm"], marker={"color": "#0099FF"}))
        st.plotly_chart(chart_layout(fig, "Weather History", height=260), use_container_width=True)


def render_what_if_lab(data: dict[str, Any], season: int) -> None:
    simulator = get_whatif_simulator()
    driver_options = sorted([(profile.full_name, driver_id) for driver_id, profile in DRIVER_DATABASE.items()])
    team_options = sorted([(profile.short_name, team_id) for team_id, profile in TEAM_DATABASE.items()])
    preset_map = {
        "Hamilton in 2023 Red Bull": ("hamilton", "red_bull", 2023),
        "Verstappen in 2025 Ferrari": ("max_verstappen", "ferrari", 2025),
        "Norris in 2021 Red Bull": ("norris", "red_bull", 2021),
        "Senna in 2023 Red Bull": ("hamilton", "red_bull", 2023),
        "Alonso in 2016 Mercedes": ("alonso", "mercedes", 2016),
    }
    preset = st.selectbox("Preset scenario", ["Custom"] + list(preset_map.keys()))
    default_driver = "hamilton"
    default_team = "red_bull"
    default_season = 2023
    if preset != "Custom":
        default_driver, default_team, default_season = preset_map[preset]

    selected_driver_name, driver_id = st.selectbox("Driver", driver_options, index=next((index for index, (_, value) in enumerate(driver_options) if value == default_driver), 0), format_func=lambda item: item[0], key="whatif_driver")
    selected_team_name, team_id = st.selectbox("Team", team_options, index=next((index for index, (_, value) in enumerate(team_options) if value == default_team), 0), format_func=lambda item: item[0], key="whatif_team")
    selected_season = st.selectbox("Season", sorted(data["race_results"]["season"].dropna().astype(int).unique().tolist(), reverse=True), index=next((index for index, value in enumerate(sorted(data["race_results"]["season"].dropna().astype(int).unique().tolist(), reverse=True)) if value == default_season), 0))
    ui_shell("What-If Lab", f"{selected_driver_name} -> {selected_team_name}", "Driver swap, championship replay, and counterfactual stress testing.")

    result = simulator.simulate_driver_swap(driver_id, team_id, int(selected_season), n_simulations=5000)
    cols = st.columns(3)
    with cols[0]:
        ui_card("Expected Points", f"{result.expected_points:.1f}", f"Actual {result.actual_points:.1f}")
    with cols[1]:
        ui_card("Expected Wins", f"{result.expected_wins:.2f}", f"Actual {result.actual_wins}")
    with cols[2]:
        ui_card("Championship Odds", f"{result.championship_probability:.0%}", f"Replaces {get_driver_name(result.replaced_driver_id)}")
    st.dataframe(result.comparison_table.head(25), use_container_width=True, height=320)

    st.markdown("#### Championship Replay")
    season_options = sorted(data["race_results"]["season"].dropna().astype(int).unique().tolist(), reverse=True)
    replay_season = st.selectbox("Replay season", season_options, key="replay_season")
    replay_round = st.selectbox("Alter round", sorted(season_calendar(data, replay_season)["round"].dropna().astype(int).tolist()), key="replay_round")
    replay_driver_name, replay_driver_id = st.selectbox("Driver to rescue", driver_options, index=0, format_func=lambda item: item[0], key="replay_driver")
    new_finish = st.slider("Override finish position", 1, 20, 5)
    replay_result = simulator.simulate_championship_scenario(
        int(replay_season),
        {"dnf_overrides": [{"round": int(replay_round), "driver_id": replay_driver_id, "new_finish_position": int(new_finish), "points": max(26 - int(new_finish), 0)}]},
    )
    st.markdown(f"<div class='f1-note'>{replay_result.narrative}</div>", unsafe_allow_html=True)
    st.dataframe(replay_result.standings.head(12), use_container_width=True, height=320)


def render_model_transparency(data: dict[str, Any], season: int) -> None:
    accuracy = aggregate_season_accuracy(season)
    ui_shell("Model Transparency", "Prediction audit and methodology", "Accuracy, calibration, feature importance, and honest model notes.")

    cols = st.columns(3)
    with cols[0]:
        ui_card("Winner Accuracy", f"{accuracy.get('season_win_accuracy', 0.0):.0%}", f"{accuracy.get('races_scored', 0)} races scored")
    with cols[1]:
        ui_card("Podium Accuracy", f"{accuracy.get('season_podium_accuracy', 0.0):.2f}/3", "Average podium hits")
    with cols[2]:
        metric_mae = data["metrics"].get("overall_mae", data["metrics"].get("mae", np.nan)) if isinstance(data["metrics"], dict) else np.nan
        ui_card("Current MAE", f"{metric_mae:.3f}" if np.isfinite(metric_mae) else "n/a", "From saved model metrics")

    calibration_paths = sorted(PROCESSED_DIR.glob("calibration_model_*.csv"))
    if calibration_paths:
        selected = st.selectbox("Calibration model", [path.stem.replace("calibration_model_", "Model ") for path in calibration_paths])
        path = next(path for path in calibration_paths if path.stem.replace("calibration_model_", "Model ") == selected)
        frame = load_dataframe(path)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=frame["prob_pred"], y=frame["prob_true"], mode="lines+markers", name="Observed"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"dash": "dash", "color": "#8A8A9A"}, name="Perfect"))
        st.plotly_chart(chart_layout(fig, "Calibration Chart", height=360), use_container_width=True)

    feature_paths = sorted(PROCESSED_DIR.glob("feature_importance_model_*.csv"))
    if feature_paths:
        selected = st.selectbox("Feature importance model", [path.stem.replace("feature_importance_model_", "Model ") for path in feature_paths])
        path = next(path for path in feature_paths if path.stem.replace("feature_importance_model_", "Model ") == selected)
        frame = load_dataframe(path).head(20)
        value_column = "gain" if "gain" in frame.columns else "importance"
        fig = go.Figure(go.Bar(x=frame[value_column], y=frame["feature"], orientation="h", marker={"color": "#E10600"}))
        st.plotly_chart(chart_layout(fig, "Feature Importance", height=420), use_container_width=True)

    st.markdown(
        """
        <div class="f1-list-card">
          <div class="f1-kicker">Methodology</div>
          <div class="f1-note">
            The platform blends standings, pace, qualifying, tyre, architecture and TrueSkill-style driver signals into race-level models.
            It is strongest once qualifying and weekend telemetry are available. It is weakest when lineups change suddenly, regulation resets are new,
            or live telemetry coverage is incomplete. Primary data sources are Jolpica, OpenF1, Open-Meteo, FastF1 cache artifacts, and the local model archive.
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


PAGES = {
    "Command Center": render_command_center,
    "Live Race": render_live_race,
    "Race Predictor": render_race_predictor,
    "Championship Intelligence": render_championship_intelligence,
    "Driver Intelligence": render_driver_intelligence,
    "Team Intelligence": render_team_intelligence,
    "Circuit Intelligence": render_circuit_intelligence,
    "What-If Lab": render_what_if_lab,
    "Model Transparency": render_model_transparency,
}


def main() -> None:
    inject_theme()
    data = load_artifacts()
    season = active_season(data)

    with st.sidebar:
        st.markdown("## F1 AI")
        st.caption("Cinematic data intelligence platform")
        season = st.selectbox("Season", sorted(data["calendar"]["season"].dropna().astype(int).unique().tolist(), reverse=True) if not data["calendar"].empty else [datetime.now().year], index=0)
        page = st.radio("Page", list(PAGES.keys()), index=0)
        coverage = data["coverage"]
        coverage_weekends = int(coverage[["season", "round"]].drop_duplicates().shape[0]) if not coverage.empty else 0
        st.markdown(f"<div class='f1-note'>OpenF1 weekends indexed: {coverage_weekends}</div>", unsafe_allow_html=True)

    PAGES[page](data, int(season))


if __name__ == "__main__":
    main()
