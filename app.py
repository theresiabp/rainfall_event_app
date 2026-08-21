from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="South Florida Rainfall Event Viewer",
    layout="wide"
)

st.title("South Florida Rainfall Event Viewer")

st.caption(
    "Explore NOAA GHCN-Daily rainfall observations and identify "
    "spatially widespread heavy-rainfall events."
)


# ============================================================
# DATA FILE
# ============================================================

DATA_FILE = Path(
    "data/processed/south_florida_ghcn.parquet"
)


# ============================================================
# CHECK THAT DATA EXISTS
# ============================================================

if not DATA_FILE.exists():

    st.error(
        "Processed rainfall data were not found.\n\n"
        "Run `python prepare_ghcn.py` first."
    )

    st.stop()


# ============================================================
# LOAD LOCAL DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_parquet(DATA_FILE)

    df["date"] = pd.to_datetime(
        df["date"]
    )

    return df


rain = load_data()


# ============================================================
# BASIC DATA INFORMATION
# ============================================================

st.sidebar.header("Event search")

st.sidebar.caption(
    f"Data period: "
    f"{rain['date'].min().date()} to "
    f"{rain['date'].max().date()}"
)


# ============================================================
# YEAR SELECTOR
# ============================================================

available_years = sorted(
    rain["date"]
    .dt.year
    .unique()
)

selected_year = st.sidebar.selectbox(
    "Year",
    available_years,
    index=len(available_years) - 1
)


# ============================================================
# MONTH SELECTOR
# ============================================================

month_names = {
    1: "January",
    2: "February",
    3: "March",
    4: "April",
    5: "May",
    6: "June",
    7: "July",
    8: "August",
    9: "September",
    10: "October",
    11: "November",
    12: "December"
}


year_data = rain[
    rain["date"].dt.year
    == selected_year
].copy()


available_months = sorted(
    year_data["date"]
    .dt.month
    .unique()
)


default_month_index = (
    available_months.index(6)
    if 6 in available_months
    else 0
)


selected_month = st.sidebar.selectbox(
    "Month",
    available_months,
    index=default_month_index,
    format_func=lambda x: month_names[x]
)


# ============================================================
# FILTER TO SELECTED MONTH
# ============================================================

month_data = year_data[
    year_data["date"].dt.month
    == selected_month
].copy()


# ============================================================
# RAINFALL THRESHOLD SLIDER
# ============================================================

threshold = st.sidebar.slider(
    "Rainfall threshold (mm/day)",
    min_value=0,
    max_value=250,
    value=50,
    step=5
)


# ============================================================
# AVAILABLE DATES
# ============================================================

available_dates = sorted(
    month_data["date"]
    .dt.date
    .unique()
)


if len(available_dates) == 0:

    st.warning(
        "No observations are available "
        "for this month."
    )

    st.stop()


# ============================================================
# DATE SLIDER
# ============================================================

selected_date = st.slider(
    "Date",
    min_value=min(available_dates),
    max_value=max(available_dates),
    value=min(available_dates),
    format="MMM DD, YYYY"
)


# ============================================================
# SELECT CURRENT DAY
# ============================================================

day = month_data[
    month_data["date"].dt.date
    == selected_date
].copy()


if day.empty:

    st.warning(
        "No station observations are available "
        "for this date."
    )

    st.stop()


# ============================================================
# THRESHOLD EXCEEDANCE
# ============================================================

day["exceed"] = (
    day["prcp_mm"]
    >= threshold
)


day["status"] = np.where(
    day["exceed"],
    f"≥ {threshold} mm",
    f"< {threshold} mm"
)


# ============================================================
# MAP MARKER SIZE
# ============================================================

# Add a small baseline so zero-rain stations
# are still visible.

day["marker_size"] = (
    5
    +
    2 * np.sqrt(
        day["prcp_mm"]
        .clip(lower=0)
    )
)

day["marker_size"] = (
    day["marker_size"]
    .clip(
        lower=5,
        upper=35
    )
)


# ============================================================
# SUMMARY METRICS
# ============================================================

n_reporting = (
    day["station"]
    .nunique()
)

n_exceeding = int(
    day["exceed"]
    .sum()
)

fraction = (
    n_exceeding
    / n_reporting
    * 100
    if n_reporting > 0
    else 0
)

max_rain = (
    day["prcp_mm"]
    .max()
)

mean_rain = (
    day["prcp_mm"]
    .mean()
)


# ============================================================
# METRIC CARDS
# ============================================================

c1, c2, c3, c4, c5 = st.columns(5)

c1.metric(
    "Reporting stations",
    n_reporting
)

c2.metric(
    f"Stations ≥ {threshold} mm",
    n_exceeding
)

c3.metric(
    "Fraction exceeding",
    f"{fraction:.1f}%"
)

c4.metric(
    "Maximum rainfall",
    f"{max_rain:.1f} mm"
)

c5.metric(
    "Mean rainfall",
    f"{mean_rain:.1f} mm"
)


# ============================================================
# MAP
# ============================================================

st.subheader(
    selected_date.strftime(
        "%B %d, %Y"
    )
)


fig = px.scatter_map(
    day,

    lat="lat",
    lon="lon",

    color="status",

    color_discrete_map={
        f"< {threshold} mm": "#9E9E9E",
        f"≥ {threshold} mm": "#D62728"
    },

    hover_name="name",

    hover_data={
        "station": True,
        "prcp_mm": ":.1f",
        "lat": ":.3f",
        "lon": ":.3f",
        "status": False
    },

    center={
        "lat": 26.2,
        "lon": -80.9
    },

    zoom=6,

    category_orders={
        "status": [
            f"< {threshold} mm",
            f"≥ {threshold} mm"
        ]
    }
)


# Same marker size for every station
fig.update_traces(
    marker=dict(
        size=10
    )
)


fig.update_layout(
    map_style="carto-positron",

    map=dict(
        center=dict(
            lat=26.2,
            lon=-80.9
        ),

        zoom=6,         
        bounds=dict(
            west=-90,
            east=-70,
            south=24,
            north=29
        )
    ),

    height=600,

    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    ),

    legend_title_text="Daily rainfall"
)

st.plotly_chart(
    fig,
    use_container_width=True
)


# ============================================================
# DAILY EVENT RANKING
# ============================================================

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

st.markdown(
    f"### Daily event ranking — "
    f"{month_names[selected_month]} "
    f"{selected_year}"
)

summary_data = month_data.copy()

summary_data["exceed"] = (
    summary_data["prcp_mm"]
    >= threshold
)

daily_summary = (
    summary_data
    .groupby("date")
    .agg(
        reporting_stations=(
            "station",
            "nunique"
        ),
        stations_exceeding=(
            "exceed",
            "sum"
        ),
        maximum_rainfall_mm=(
            "prcp_mm",
            "max"
        ),
        mean_rainfall_mm=(
            "prcp_mm",
            "mean"
        )
    )
    .reset_index()
)

daily_summary["fraction_exceeding"] = (
    daily_summary["stations_exceeding"]
    /
    daily_summary["reporting_stations"]
    * 100
)

daily_summary = (
    daily_summary
    .sort_values(
        [
            "stations_exceeding",
            "maximum_rainfall_mm"
        ],
        ascending=False
    )
    .reset_index(drop=True)
)

daily_summary["maximum_rainfall_mm"] = (
    daily_summary["maximum_rainfall_mm"]
    .round(1)
)

daily_summary["mean_rainfall_mm"] = (
    daily_summary["mean_rainfall_mm"]
    .round(1)
)

daily_summary["fraction_exceeding"] = (
    daily_summary["fraction_exceeding"]
    .round(1)
)

daily_summary = (
    daily_summary.rename(
        columns={
            "date": "Date",
            "reporting_stations": "Reporting stations",
            "stations_exceeding": f"Stations ≥ {threshold} mm",
            "fraction_exceeding": "% exceeding",
            "maximum_rainfall_mm": "Maximum rainfall (mm)",
            "mean_rainfall_mm": "Mean rainfall (mm)"
        }
    )
)

# format date nicely
daily_summary["Date"] = pd.to_datetime(
    daily_summary["Date"]
).dt.strftime("%Y-%m-%d")

st.dataframe(
    daily_summary,
    use_container_width=True,
    hide_index=True
)

# ============================================================
# CURRENT-DAY STATION TABLE
# ============================================================

with st.expander(
    "Show station observations"
):

    station_table = (
        day[
            [
                "station",
                "name",
                "prcp_mm",
                "lat",
                "lon",
                "exceed"
            ]
        ]
        .sort_values(
            "prcp_mm",
            ascending=False
        )
    )


    station_table = (
        station_table.rename(
            columns={
                "station":
                    "Station ID",

                "name":
                    "Station",

                "prcp_mm":
                    "Rainfall (mm)",

                "lat":
                    "Latitude",

                "lon":
                    "Longitude",

                "exceed":
                    "Exceeds threshold"
            }
        )
    )


    st.dataframe(
        station_table,
        use_container_width=True,
        hide_index=True
    )