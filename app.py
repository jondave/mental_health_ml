from __future__ import annotations

import base64
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
import folium
from branca.colormap import linear
from flask import Flask, render_template, request
from flask_socketio import SocketIO
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")


def infer_pollutant_from_filename(file_name: str) -> str:
    name = file_name.upper()
    if "NO2" in name:
        return "NO2"
    if "PM10" in name:
        return "PM10"
    if "PM25" in name or "PM2.5" in name:
        return "PM2.5"
    return "UNKNOWN"


def parse_24h_timestamp(date_series: pd.Series, hour_series: pd.Series) -> pd.Series:
    d = pd.to_datetime(date_series, format="%d-%m-%Y", errors="coerce")
    h = hour_series.astype(str).str.strip()
    is_24 = h.eq("24:00")
    h = h.where(~is_24, "00:00")
    dt = pd.to_datetime(
        d.dt.strftime("%Y-%m-%d") + " " + h,
        format="%Y-%m-%d %H:%M",
        errors="coerce",
    )
    dt = dt.where(~is_24, dt + pd.Timedelta(days=1))
    return dt


def locate_dataset_dir() -> Path:
    here = Path(__file__).resolve().parent
    candidates = [
        here / "datasets" / "arun",
        Path("datasets/arun").resolve(),
        Path("code/mental_health_ml/datasets/arun").resolve(),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError("Could not locate datasets/arun directory.")


def load_aurn_data() -> pd.DataFrame:
    base_dir = locate_dataset_dir()
    site_label_map = {
        "immingham": "Immingham Woodlands",
        "lincoln": "Lincoln Canwick Road",
        "scunthorpe": "Scunthorpe Town",
        "tallington": "Tallington",
        "toft_newton": "Toft Newton",
    }

    records: List[pd.DataFrame] = []
    files_found = sorted(base_dir.rglob("*.csv"))

    for csv_file in files_found:
        pollutant = infer_pollutant_from_filename(csv_file.name)
        site_folder = csv_file.parent.name.lower()
        site_name = site_label_map.get(site_folder, csv_file.parent.name)

        raw = pd.read_csv(csv_file, skiprows=4, dtype=str)
        raw.columns = [str(c).strip() for c in raw.columns]
        date_col = raw.columns[0]
        raw = raw.rename(columns={date_col: "Date"})

        hour_cols = [
            c
            for c in raw.columns
            if isinstance(c, str) and ":" in c and len(c.strip()) == 5
        ]
        if not hour_cols:
            continue

        long_df = raw[["Date"] + hour_cols].melt(
            id_vars="Date",
            var_name="Hour",
            value_name="Value",
        )

        long_df["Value"] = long_df["Value"].astype(str).str.strip()
        long_df["Value"] = long_df["Value"].replace({"": np.nan, "nan": np.nan})
        long_df["Value"] = pd.to_numeric(long_df["Value"], errors="coerce")
        long_df["Timestamp"] = parse_24h_timestamp(long_df["Date"], long_df["Hour"])
        long_df = long_df.dropna(subset=["Timestamp", "Value"])

        long_df["Site_Name"] = site_name
        long_df["Pollutant"] = pollutant
        records.append(long_df[["Timestamp", "Site_Name", "Pollutant", "Value"]])

    if not records:
        raise ValueError("No valid AURN records loaded. Check CSV file contents.")

    df_aurn = pd.concat(records, ignore_index=True).sort_values("Timestamp")
    return df_aurn.reset_index(drop=True)


def build_hourly_wide(df_aurn: pd.DataFrame) -> pd.DataFrame:
    df_wide = (
        df_aurn.groupby(["Timestamp", "Site_Name", "Pollutant"], as_index=False)["Value"]
        .mean()
        .pivot(index=["Timestamp", "Site_Name"], columns="Pollutant", values="Value")
        .reset_index()
    )
    for col in ["NO2", "PM10", "PM2.5"]:
        if col not in df_wide.columns:
            df_wide[col] = np.nan
    return df_wide


def simulate_people(df_wide: pd.DataFrame, n: int = 1000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    site_names = [
        "Lincoln Canwick Road",
        "Scunthorpe Town",
        "Immingham Woodlands",
        "Toft Newton",
        "Tallington",
    ]
    site_probs = np.array([0.24, 0.22, 0.18, 0.18, 0.18])
    site_to_environment = {
        "Lincoln Canwick Road": "Urban",
        "Scunthorpe Town": "Urban",
        "Immingham Woodlands": "Urban",
        "Toft Newton": "Rural",
        "Tallington": "Rural",
    }
    site_to_ons = {
        "Lincoln Canwick Road": "Urban city and town",
        "Scunthorpe Town": "Urban city and town",
        "Immingham Woodlands": "Urban city and town",
        "Toft Newton": "Rural hamlets and isolated dwellings",
        "Tallington": "Rural town and fringe",
    }

    pool = df_wide[["Timestamp", "Site_Name", "NO2", "PM2.5", "PM10"]].copy()
    pool = pool[pool["Site_Name"].isin(site_names)].copy()
    pool["Environment_Type"] = pool["Site_Name"].map(site_to_environment)
    pool["Month"] = pool["Timestamp"].dt.month
    season_map = {
        12: "Winter", 1: "Winter", 2: "Winter",
        3: "Spring", 4: "Spring", 5: "Spring",
        6: "Summer", 7: "Summer", 8: "Summer",
        9: "Autumn", 10: "Autumn", 11: "Autumn",
    }
    pool["Season"] = pool["Month"].map(season_map)

    for pol in ["NO2", "PM2.5", "PM10"]:
        by_env_month = pool.groupby(["Environment_Type", "Month"])[pol].transform("mean")
        by_env = pool.groupby("Environment_Type")[pol].transform("mean")
        pool[pol] = pool[pol].fillna(by_env_month).fillna(by_env).fillna(pool[pol].mean())

    selected_sites = rng.choice(site_names, size=n, p=site_probs)
    urban_rural_status = np.array([site_to_ons[s] for s in selected_sites])

    site_profiles = {
        "Lincoln Canwick Road": {"min": 1, "max": 8, "peak": 2.0},
        "Scunthorpe Town": {"min": 1, "max": 5, "peak": 2.0},
        "Immingham Woodlands": {"min": 1, "max": 7, "peak": 3.0},
        "Toft Newton": {"min": 6, "max": 10, "peak": 8.5},
        "Tallington": {"min": 7, "max": 10, "peak": 7.5},
    }

    imd_decile = np.array([
        int(np.clip(np.round(rng.triangular(site_profiles[s]["min"], site_profiles[s]["peak"], site_profiles[s]["max"])), 1, 10))
        for s in selected_sites
    ])
    age = np.clip(rng.normal(41, 16, n).round().astype(int), 18, 90)

    exp_no2 = np.full(n, np.nan)
    exp_pm25 = np.full(n, np.nan)
    exp_pm10 = np.full(n, np.nan)
    exp_month = np.zeros(n, dtype=int)
    exp_season = np.empty(n, dtype=object)

    for site in site_names:
        idx = np.where(selected_sites == site)[0]
        k = len(idx)
        if k == 0:
            continue
        site_pool = pool[pool["Site_Name"] == site]
        sampled = site_pool.sample(n=k, replace=True, random_state=int(rng.integers(0, 1_000_000))).reset_index(drop=True)
        exp_no2[idx] = sampled["NO2"].to_numpy()
        exp_pm25[idx] = sampled["PM2.5"].to_numpy()
        exp_pm10[idx] = sampled["PM10"].to_numpy()
        exp_month[idx] = sampled["Month"].to_numpy()
        exp_season[idx] = sampled["Season"].to_numpy()

    site_random_effect_map = {
        "Lincoln Canwick Road": 0.35,
        "Scunthorpe Town": 0.25,
        "Immingham Woodlands": 0.20,
        "Toft Newton": -0.10,
        "Tallington": -0.05,
    }
    site_random_effect = np.array([site_random_effect_map[s] for s in selected_sites])
    comorbidity_risk = np.clip(0.12 + 0.005 * age + 0.03 * (imd_decile - 1) + rng.normal(0, 0.08, n), 0, 1)

    return pd.DataFrame({
        "Site_Name": selected_sites,
        "Age": age,
        "IMD_Decile": imd_decile,
        "Urban_Rural_Status": urban_rural_status,
        "Month": exp_month,
        "Season": exp_season,
        "Comorbidity_Risk": comorbidity_risk,
        "Site_Random_Effect": site_random_effect,
        "NO2_Exposure": np.clip(exp_no2, 1.0, None).round(3),
        "PM25_Exposure": np.clip(exp_pm25, 1.0, None).round(3),
        "PM10_Exposure": np.clip(exp_pm10, 1.0, None).round(3),
    })


def apply_target(df: pd.DataFrame, weights: Dict[str, float], noise_std: float = 1.8, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    urban_flag = df["Urban_Rural_Status"].str.startswith("Urban").astype(int)
    winter_flag = (df["Season"] == "Winter").astype(int)

    expected = (
        weights["intercept"]
        + weights["no2"] * df["NO2_Exposure"]
        + weights["pm25"] * df["PM25_Exposure"]
        + weights["pm10"] * df["PM10_Exposure"]
        + weights["imd"] * df["IMD_Decile"]
        + weights["urban"] * urban_flag
        + weights["age"] * df["Age"]
        + weights["comorbidity"] * df["Comorbidity_Risk"]
        + weights["site_effect"] * df["Site_Random_Effect"]
        + weights["winter"] * winter_flag
        + weights["pm25_urban_interaction"] * df["PM25_Exposure"] * urban_flag
        + weights["no2_urban_interaction"] * df["NO2_Exposure"] * urban_flag
    )
    phq9 = expected + rng.normal(0, noise_std, size=len(df))
    out = df.copy()
    out["PHQ9"] = np.clip(np.round(phq9), 0, 27).astype(int)
    return out


def fig_to_base64() -> str:
    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=140, bbox_inches="tight")
    plt.close()
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")


def make_correlation_plot(df: pd.DataFrame) -> str:
    corr_df = df.copy()
    corr_df["Is_Urban"] = np.where(corr_df["Urban_Rural_Status"].str.startswith("Urban"), 1, 0)
    corr_df["Is_Winter"] = (corr_df["Season"] == "Winter").astype(int)
    features = [
        "Age", "IMD_Decile", "Comorbidity_Risk", "NO2_Exposure", "PM25_Exposure",
        "PM10_Exposure", "Is_Urban", "Is_Winter", "PHQ9",
    ]
    corr = corr_df[features].corr(numeric_only=True)

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
    )
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    return fig_to_base64()


def train_model(df: pd.DataFrame) -> Tuple[RandomForestRegressor, pd.DataFrame, pd.Series, np.ndarray, Dict[str, float]]:
    model_df = df.copy()
    model_df["Is_Urban"] = np.where(model_df["Urban_Rural_Status"].str.startswith("Urban"), 1, 0)
    model_df["NO2"] = model_df["NO2_Exposure"]
    model_df["PM25"] = model_df["PM25_Exposure"]
    model_df["PM10"] = model_df["PM10_Exposure"]

    feature_cols = ["Age", "IMD_Decile", "NO2", "PM25", "PM10", "Is_Urban"]
    X = model_df[feature_cols]
    y = model_df["PHQ9"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "r2": float(r2_score(y_test, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
        "mae": float(mean_absolute_error(y_test, pred)),
    }
    return model, X_test, y_test, pred, metrics


def make_shap_plots(model: RandomForestRegressor, X_test: pd.DataFrame) -> Dict[str, str]:
    sample = X_test.sample(n=min(len(X_test), 250), random_state=42)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    shap.summary_plot(shap_values, sample, show=False)
    plt.title("SHAP Summary")
    plt.tight_layout()
    summary_img = fig_to_base64()

    shap.summary_plot(shap_values, sample, plot_type="bar", show=False)
    plt.title("SHAP Feature Importance")
    plt.tight_layout()
    bar_img = fig_to_base64()

    return {
        "summary": summary_img,
        "bar": bar_img,
    }


def make_urban_rural_plot(X_test: pd.DataFrame, y_test: pd.Series, pred: np.ndarray) -> str:
    results_df = pd.DataFrame({
        "Area_Type": np.where(X_test["Is_Urban"] == 1, "Urban", "Rural"),
        "Actual_PHQ9": y_test.to_numpy(),
        "Predicted_PHQ9": pred,
    })
    results_long = results_df.melt(
        id_vars="Area_Type",
        value_vars=["Actual_PHQ9", "Predicted_PHQ9"],
        var_name="Series",
        value_name="PHQ9_Score",
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1.7, 1]})
    sns.boxplot(
        data=results_long,
        x="Area_Type",
        y="PHQ9_Score",
        hue="Series",
        palette={"Actual_PHQ9": "#4c78a8", "Predicted_PHQ9": "#f58518"},
        ax=axes[0],
    )
    axes[0].set_title("Actual vs Predicted PHQ-9 by Area Type")
    axes[0].set_xlabel("Area Type")
    axes[0].set_ylabel("PHQ-9 Score")
    axes[0].legend(title="Series")

    mean_compare = (
        results_long
        .groupby(["Area_Type", "Series"], as_index=False)["PHQ9_Score"]
        .mean()
    )
    sns.pointplot(
        data=mean_compare,
        x="Area_Type",
        y="PHQ9_Score",
        hue="Series",
        palette={"Actual_PHQ9": "#4c78a8", "Predicted_PHQ9": "#f58518"},
        dodge=0.25,
        linestyle="none",
        markers=["o", "D"],
        ax=axes[1],
    )
    axes[1].set_title("Mean Actual vs Predicted PHQ-9")
    axes[1].set_xlabel("Area Type")
    axes[1].set_ylabel("Mean PHQ-9 Score")
    axes[1].legend(title="Series")

    plt.suptitle("Urban vs Rural Comparison from Random Forest Test Results", y=1.03)
    plt.tight_layout()
    return fig_to_base64()


def make_district_map(df_model: pd.DataFrame, model: RandomForestRegressor) -> Tuple[str, str]:
    district_specs = [
        {"District": "Lincoln", "Latitude": 53.2307, "Longitude": -0.5406, "Source_Site": "Lincoln Canwick Road", "Is_Urban": 1},
        {"District": "North Lincolnshire (Scunthorpe)", "Latitude": 53.5865, "Longitude": -0.6544, "Source_Site": "Scunthorpe Town", "Is_Urban": 1},
        {"District": "North East Lincolnshire (Grimsby)", "Latitude": 53.5654, "Longitude": -0.0808, "Source_Site": "Immingham Woodlands", "Is_Urban": 1},
        {"District": "West Lindsey", "Latitude": 53.3460, "Longitude": -0.6600, "Source_Site": "Toft Newton", "Is_Urban": 0},
        {"District": "East Lindsey (Skegness)", "Latitude": 53.1439, "Longitude": 0.3400, "Source_Site": "Immingham Woodlands", "Is_Urban": 0},
        {"District": "Boston", "Latitude": 52.9789, "Longitude": -0.0237, "Source_Site": "Scunthorpe Town", "Is_Urban": 1},
        {"District": "North Kesteven", "Latitude": 53.0996, "Longitude": -0.1900, "Source_Site": "Lincoln Canwick Road", "Is_Urban": 0},
        {"District": "South Kesteven", "Latitude": 52.9112, "Longitude": -0.6418, "Source_Site": "Tallington", "Is_Urban": 0},
        {"District": "South Holland", "Latitude": 52.7867, "Longitude": -0.1517, "Source_Site": "Tallington", "Is_Urban": 0},
    ]

    global_medians = {
        "Age": float(df_model["Age"].median()),
        "IMD_Decile": float(df_model["IMD_Decile"].median()),
        "NO2": float(df_model["NO2_Exposure"].median()),
        "PM25": float(df_model["PM25_Exposure"].median()),
        "PM10": float(df_model["PM10_Exposure"].median()),
    }

    rows: List[Dict[str, float | int | str]] = []
    for spec in district_specs:
        site_slice = df_model[df_model["Site_Name"] == spec["Source_Site"]]
        if len(site_slice) == 0:
            features = global_medians.copy()
        else:
            features = {
                "Age": float(site_slice["Age"].median()),
                "IMD_Decile": float(site_slice["IMD_Decile"].median()),
                "NO2": float(site_slice["NO2_Exposure"].median()),
                "PM25": float(site_slice["PM25_Exposure"].median()),
                "PM10": float(site_slice["PM10_Exposure"].median()),
            }

        rows.append({
            "District": spec["District"],
            "Latitude": spec["Latitude"],
            "Longitude": spec["Longitude"],
            "Age": features["Age"],
            "IMD_Decile": features["IMD_Decile"],
            "NO2": features["NO2"],
            "PM25": features["PM25"],
            "PM10": features["PM10"],
            "Is_Urban": int(spec["Is_Urban"]),
            "Source_Site": spec["Source_Site"],
        })

    region_df = pd.DataFrame(rows)
    feature_cols = ["Age", "IMD_Decile", "NO2", "PM25", "PM10", "Is_Urban"]
    region_df["Predicted_PHQ9"] = model.predict(region_df[feature_cols]).round(2)

    district_map = folium.Map(location=[53.2, -0.2], zoom_start=9, tiles="CartoDB positron")
    score_min = float(region_df["Predicted_PHQ9"].min())
    score_max = float(region_df["Predicted_PHQ9"].max())
    color_scale = linear.YlOrRd_09.scale(score_min, score_max)
    color_scale.caption = "Predicted PHQ-9 Score"

    for _, row in region_df.iterrows():
        score = float(row["Predicted_PHQ9"])
        folium.CircleMarker(
            location=[row["Latitude"], row["Longitude"]],
            radius=40,
            color=color_scale(score),
            fill=True,
            fill_color=color_scale(score),
            fill_opacity=0.55,
            weight=2,
            popup=(
                f"<b>{row['District']}</b><br>"
                f"Predicted PHQ-9: {score:.2f}<br>"
            ),
        ).add_to(district_map)

    color_scale.add_to(district_map)
    table_df = region_df[["District", "Source_Site", "Is_Urban", "Predicted_PHQ9"]].copy()
    table_df["Is_Urban"] = table_df["Is_Urban"].map({1: "Urban", 0: "Rural"})
    table_df = table_df.rename(columns={"Is_Urban": "Urban/Rural"})
    district_table_html = table_df.sort_values("Predicted_PHQ9", ascending=False).to_html(
        index=False, classes="district-table"
    )

    return district_map._repr_html_(), district_table_html


def default_weights() -> Dict[str, float]:
    return {
        "intercept": 1.4,
        "no2": 0.19,
        "pm25": 0.20,
        "pm10": 0.17,
        "imd": -0.52,
        "urban": 0.55,
        "age": -0.010,
        "comorbidity": 2.2,
        "site_effect": 1.4,
        "winter": 0.45,
        "pm25_urban_interaction": 0.045,
        "no2_urban_interaction": 0.080,
    }


def parse_weights(form_data) -> Dict[str, float]:
    defaults = default_weights()
    out: Dict[str, float] = {}
    for key, value in defaults.items():
        out[key] = float(form_data.get(key, value))
    return out


@app.route("/", methods=["GET", "POST"])
def index():
    weights = default_weights()
    metrics = None
    images = None
    error = None

    if request.method == "POST":
        try:
            weights = parse_weights(request.form)
            df_aurn = load_aurn_data()
            df_wide = build_hourly_wide(df_aurn)
            df_people = simulate_people(df_wide, n=1000, seed=42)
            df_model = apply_target(df_people, weights=weights, noise_std=1.8, seed=42)

            corr_img = make_correlation_plot(df_model)
            model, X_test, y_test, pred, metrics = train_model(df_model)
            shap_imgs = make_shap_plots(model, X_test)
            urban_rural_img = make_urban_rural_plot(X_test, y_test, pred)
            folium_map_html, district_table_html = make_district_map(df_model, model)

            images = {
                "corr": corr_img,
                "shap_summary": shap_imgs["summary"],
                "shap_bar": shap_imgs["bar"],
                "urban_rural": urban_rural_img,
                "district_map": folium_map_html,
                "district_table": district_table_html,
            }
        except Exception as exc:  # broad for UI friendliness
            error = str(exc)

    return render_template(
        "index.html",
        weights=weights,
        metrics=metrics,
        images=images,
        error=error,
    )


if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5005, debug=True)
