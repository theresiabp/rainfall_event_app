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
    page_icon="🌧️",
    layout="wide"
)

st.title("🌧️ South Florida Rainfall Event Viewer")

st.caption(
    "Explore daily GHCN rainfall observations and identify "
    "spatially widespread heavy-rainfall events."
)


# ============================================================
# DATA FILES
# ============================================================

DATA_FILE = Path(
    "data/processed/south_florida_ghcn.parquet"
)

ENSO_FILE = Path(
    "data/processed/nino34_daily.parquet"
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


if not ENSO_FILE.exists():

    st.error(
        "Processed Niño 3.4 SST data were not found.\n\n"
        "Create `data/processed/nino34_daily.parquet` first."
    )

    st.stop()


# ============================================================
# LOAD LOCAL DATA
# ============================================================

@st.cache_data
def load_data():

    df = pd.read_parquet(
        DATA_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    return df


@st.cache_data
def load_enso_data():

    df = pd.read_parquet(
        ENSO_FILE
    )

    df["date"] = pd.to_datetime(
        df["date"]
    ).dt.normalize()

    return df


rain = load_data()
enso = load_enso_data()


# ============================================================
# BASIC DATA INFORMATION
# ============================================================

st.sidebar.header(
    "Event search"
)

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
# UNIT SELECTOR
# ============================================================

unit = st.sidebar.radio(
    "Rainfall units",
    ["mm", "inches"],
    horizontal=True
)


if unit == "mm":

    rainfall_column = "prcp_mm"
    unit_label = "mm"

    threshold_display = st.sidebar.slider(
        "Rainfall threshold (mm/day)",
        min_value=0.0,
        max_value=250.0,
        value=50.0,
        step=5.0
    )

else:

    rainfall_column = "prcp_in"
    unit_label = "in"

    threshold_display = st.sidebar.slider(
        "Rainfall threshold (in/day)",
        min_value=0.0,
        max_value=10.0,
        value=2.0,
        step=0.1
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

st.subheader(
    "Explore the event"
)

selected_date = st.slider(
    "Date",
    min_value=min(
        available_dates
    ),
    max_value=max(
        available_dates
    ),
    value=min(
        available_dates
    ),
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
# GET NINO 3.4 SST DATA FOR SELECTED DATE
# ============================================================

selected_date_pd = pd.Timestamp(
    selected_date
).normalize()

enso_row = enso.loc[
    enso["date"] == selected_date_pd
]

if not enso_row.empty:

    enso_ltm = (
        enso_row["ltm_1991_2020"]
        .iloc[0]
    )

    enso_anomaly = (
        enso_row["anomaly"]
        .iloc[0]
    )

else:

    enso_ltm = np.nan
    enso_anomaly = np.nan


# ============================================================
# THRESHOLD EXCEEDANCE
# ============================================================

day["exceed"] = (
    day[rainfall_column]
    >= threshold_display
)

day["status"] = np.where(
    day["exceed"],
    f"≥ {threshold_display:g} {unit_label}",
    f"< {threshold_display:g} {unit_label}"
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
    day[rainfall_column]
    .max()
)

mean_rain = (
    day[rainfall_column]
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
    f"Stations ≥ "
    f"{threshold_display:g} "
    f"{unit_label}",
    n_exceeding
)

c3.metric(
    "Fraction exceeding",
    f"{fraction:.1f}%"
)

c4.metric(
    "Maximum rainfall",
    f"{max_rain:.2f} "
    f"{unit_label}"
)

c5.metric(
    "Mean rainfall",
    f"{mean_rain:.2f} "
    f"{unit_label}"
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
        f"< {threshold_display:g} {unit_label}":
            "#9E9E9E",

        f"≥ {threshold_display:g} {unit_label}":
            "#D62728"
    },

    hover_name="name",

    hover_data={
        "station": True,
        rainfall_column: ":.2f",
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
            f"< {threshold_display:g} {unit_label}",
            f"≥ {threshold_display:g} {unit_label}"
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
    map_style="open-street-map",

    height=700,

    margin=dict(
        l=0,
        r=0,
        t=0,
        b=0
    ),

    legend_title_text=(
        "Daily rainfall"
    )
)


# ============================================================
# MAP + ENSO CARD
# ============================================================

map_col, enso_col = st.columns(
    [5, 1.3],
    gap="medium"
)


with map_col:

    st.plotly_chart(
        fig,
        use_container_width=True
    )


with enso_col:

    st.markdown(
        "### ENSO"
    )

    st.caption(
        "Niño 3.4"
    )

    if pd.notna(enso_anomaly):

        st.metric(
            "SST anomaly",
            f"{enso_anomaly:+.2f} °C"
        )

        st.metric(
            "1991–2020 LTM",
            f"{enso_ltm:.2f} °C"
        )

    else:

        st.info(
            "Niño 3.4 SST data are unavailable "
            "for this date."
        )


# ============================================================
# DAILY EVENT RANKING
# ============================================================

st.markdown(
    "<br>",
    unsafe_allow_html=True
)

st.divider()

st.markdown(
    f"### Daily event ranking — "
    f"{month_names[selected_month]} "
    f"{selected_year}"
)


summary_data = (
    month_data.copy()
)

summary_data["exceed"] = (
    summary_data[rainfall_column]
    >= threshold_display
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

        maximum_rainfall=(
            rainfall_column,
            "max"
        ),

        mean_rainfall=(
            rainfall_column,
            "mean"
        )
    )
    .reset_index()
)


daily_summary[
    "fraction_exceeding"
] = (
    daily_summary[
        "stations_exceeding"
    ]
    /
    daily_summary[
        "reporting_stations"
    ]
    * 100
)


daily_summary = (
    daily_summary
    .sort_values(
        [
            "stations_exceeding",
            "maximum_rainfall"
        ],
        ascending=False
    )
    .reset_index(
        drop=True
    )
)


daily_summary[
    "maximum_rainfall"
] = (
    daily_summary[
        "maximum_rainfall"
    ]
    .round(2)
)


daily_summary[
    "mean_rainfall"
] = (
    daily_summary[
        "mean_rainfall"
    ]
    .round(2)
)


daily_summary[
    "fraction_exceeding"
] = (
    daily_summary[
        "fraction_exceeding"
    ]
    .round(1)
)


daily_summary = (
    daily_summary.rename(
        columns={
            "date":
                "Date",

            "reporting_stations":
                "Reporting stations",

            "stations_exceeding":
                f"Stations ≥ "
                f"{threshold_display:g} "
                f"{unit_label}",

            "fraction_exceeding":
                "% exceeding",

            "maximum_rainfall":
                f"Maximum rainfall "
                f"({unit_label})",

            "mean_rainfall":
                f"Mean rainfall "
                f"({unit_label})"
        }
    )
)


daily_summary["Date"] = (
    pd.to_datetime(
        daily_summary["Date"]
    )
    .dt.strftime(
        "%Y-%m-%d"
    )
)


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
                rainfall_column,
                "lat",
                "lon",
                "exceed"
            ]
        ]
        .sort_values(
            rainfall_column,
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

                rainfall_column:
                    f"Rainfall ({unit_label})",

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
