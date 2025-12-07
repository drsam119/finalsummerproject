import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# ------------------------------------------------------------
# PAGE SETTINGS
# ------------------------------------------------------------
st.set_page_config(
    page_title="Employment Data Dashboard and Machine Learning Forecast",
    layout="wide",
)


# ------------------------------------------------------------
# DATA LOADING
# ------------------------------------------------------------
# CSV must be in the same folder as this script
df = pd.read_csv("combinedemploymentdatafinal_with_years.csv")

# Clean and prepare state names
df["State Name"] = df["State Name"].astype(str).str.strip()

snm = {
    "Delhi": "NCT of Delhi",
    "Pondicherry": "Puducherry",
    "Odisha": "Orissa",
}
df["State Name"] = df["State Name"].replace(snm)

# For ML we also keep a 'State' column
df["State"] = df["State Name"]

# Sector columns (used in charts AND ML)
sector_cols = ["Service_Sector(%)", "Manufacturing_Sector(%)", "Agriculture_Sector(%)"]

# Overall Employability
df["Overall Employability (%)"] = df[
    [
        "Urban_Employability(%)",
        "Rural_Employability(%)",
        "Male_Employability(%)",
        "Female_Employability(%)",
    ]
].mean(axis=1)

st.title("Employment Data Dashboard and Machine Learning Forecast")


# ------------------------------------------------------------
# CHARTS (VISUALISATIONS)
# ------------------------------------------------------------

# Chart 1: Urban vs Rural
regionforbar = (
    df.groupby("Region")[["Urban_Employability(%)", "Rural_Employability(%)"]]
    .mean()
    .reset_index()
)

fig1 = px.bar(
    regionforbar,
    x="Region",
    y=["Urban_Employability(%)", "Rural_Employability(%)"],
    title="Comparison of Urban and Rural Employability by Region",
    barmode="group",
    labels={"value": "Employability (%)", "variable": "Employment Type"},
)

# Chart 2: Graduate % (pie)
regionforpie = df.groupby("Region")[["Graduate(%)"]].mean().reset_index()

fig2 = px.pie(
    regionforpie,
    values="Graduate(%)",
    names="Region",
    title="Average Percentage of Graduates by Region",
)

# Chart 3: Age Groups
agegroupcomp = (
    df.groupby("Region")[["Age_Group_18_25(%)", "Age_Group_26_35(%)"]]
    .mean()
    .reset_index()
)

agegroupmelt = agegroupcomp.melt(
    id_vars="Region", var_name="Age Group", value_name="Average Employability (%)"
)

fig3 = px.bar(
    agegroupmelt,
    x="Region",
    y="Average Employability (%)",
    color="Age Group",
    title="Average Employability by Age Group and Region",
    barmode="group",
)

# Chart 4: Gender
fig4 = go.Figure(
    data=[
        go.Bar(
            name="Male Employability",
            x=df["State Name"],
            y=df["Male_Employability(%)"],
        ),
        go.Bar(
            name="Female Employability",
            x=df["State Name"],
            y=df["Female_Employability(%)"],
        ),
    ]
)
fig4.update_layout(
    title="Gender-wise Employability by State/UT",
    xaxis_title="State/UT",
    yaxis_title="Employability (%)",
    barmode="group",
)

# Chart 5: Sector-wise
sectoremp = df[["State Name"] + sector_cols]

sectorempmelt = sectoremp.melt(
    id_vars="State Name", var_name="Sector", value_name="Employability (%)"
)

fig5 = px.bar(
    sectorempmelt,
    x="State Name",
    y="Employability (%)",
    color="Sector",
    title="Sector-wise Employability by State/UT",
)

fig5.update_layout(
    xaxis_title="State/UT",
    yaxis_title="Employability (%)",
    barmode="stack",
)

# Chart 6: Qualification Distribution (SUNBURST)
qualification = df.melt(
    id_vars="State Name",
    value_vars=["Graduate(%)", "ITI(%)", "Diploma(%)"],
    var_name="Qualification",
    value_name="Percentage",
)

fig6 = px.sunburst(
    qualification,
    path=["State Name", "Qualification"],
    values="Percentage",
    title="Qualification Distribution by State/UT",
)

# Chart 7: 3D BAR MAP (replacement for choropleth)
# Use latest year so each state has one value
if "Year" in df.columns:
    latest_year = df["Year"].max()
    df_latest = df[df["Year"] == latest_year]
else:
    latest_year = None
    df_latest = df

overall_by_state = (
    df_latest.groupby("State", as_index=False)["Overall Employability (%)"]
    .mean()
)

# Build a 3D "bar" using vertical line segments (Scatter3d)
x_positions = list(range(len(overall_by_state)))
state_names = overall_by_state["State"].tolist()
z_heights = overall_by_state["Overall Employability (%)"].tolist()

xs = []
ys = []
zs = []

for i, z in enumerate(z_heights):
    # vertical line from z=0 to z=height at x=i, y=0
    xs += [x_positions[i], x_positions[i], None]
    ys += [0, 0, None]
    zs += [0, z, None]

fig7 = go.Figure(
    data=[
        go.Scatter3d(
            x=xs,
            y=ys,
            z=zs,
            mode="lines+markers",
            line=dict(width=6),
            marker=dict(size=4),
        )
    ]
)

fig7.update_layout(
    title=(
        "3D Bar Map of Overall Employability by State/UT"
        + (f" (Year {latest_year})" if latest_year is not None else "")
    ),
    scene=dict(
        xaxis=dict(
            title="State/UT",
            tickmode="array",
            tickvals=x_positions,
            ticktext=state_names,
        ),
        yaxis=dict(title="Axis", showticklabels=False),
        zaxis=dict(title="Overall Employability (%)"),
    ),
    margin=dict(l=0, r=0, b=0, t=40),
)

# NEW: Education Impact Scatter Plot (with trendline)
fig_edu_scatter = px.scatter(
    df,
    x="Graduate(%)",
    y="Overall Employability (%)",
    color="Region",
    hover_name="State Name",
    size="Graduate(%)",
    title="Impact of Education on Overall Employability",
    labels={
        "Graduate(%)": "Percentage of Graduates",
        "Overall Employability (%)": "Overall Employability (%)",
    },
    trendline="ols",
)


# ------------------------------------------------------------
# SIDEBAR SELECTION
# ------------------------------------------------------------
chartcategories = {
    "Demographics": [
        "Urban Employability",
        "Rural Employability",
        "Age Group Employability",
        "Overall Employability",
    ],
    "Education": [
        "Education Impact (Scatter)",
        "Qualification Distribution",
    ],
    "Industry": [
        "Sector-wise Employability",
    ],
}

selected_category = st.sidebar.selectbox(
    "Select Category", list(chartcategories.keys())
)
selected_chart = st.sidebar.selectbox(
    "Select Chart", chartcategories[selected_category]
)

if selected_chart == "Urban Employability":
    st.plotly_chart(fig1, use_container_width=True)

elif selected_chart == "Rural Employability":
    st.plotly_chart(fig2, use_container_width=True)

elif selected_chart == "Age Group Employability":
    st.plotly_chart(fig3, use_container_width=True)

elif selected_chart == "Graduate Employability":
    # If you still keep this label somewhere, use the pie:
    st.plotly_chart(fig2, use_container_width=True)

elif selected_chart == "Sector-wise Employability":
    st.plotly_chart(fig5, use_container_width=True)

elif selected_chart == "Qualification Distribution":
    st.plotly_chart(fig6, use_container_width=True)

elif selected_chart == "Overall Employability":
    st.plotly_chart(fig7, use_container_width=True)

elif selected_chart == "Education Impact (Scatter)":
    st.plotly_chart(fig_edu_scatter, use_container_width=True)


# ------------------------------------------------------------
# MACHINE LEARNING FORECASTING (AI/ML SECTION)
# ------------------------------------------------------------
st.markdown("---")
st.header("Machine Learning Forecasting")

# Long-format data for ML
df_long = df.melt(
    id_vars=["State", "Region", "Year"],
    value_vars=sector_cols,
    var_name="Sector",
    value_name="Employment",
)

unique_states = sorted(df_long["State"].unique())
unique_sectors = sorted(df_long["Sector"].unique())
years = sorted(df_long["Year"].unique())

has_time_series = len(years) >= 2

if not has_time_series:
    st.warning("Multiple distinct years are required for forecasting.")
else:
    # 1. Global RandomForest model (for overall performance metrics only)
    X = df_long[["State", "Region", "Year", "Sector"]]
    y = df_long["Employment"]

    cat_cols = ["State", "Region", "Sector"]
    num_cols = ["Year"]

    preprocess = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )

    model = Pipeline(
        steps=[
            ("preprocess", preprocess),
            ("regressor", RandomForestRegressor(n_estimators=200, random_state=42)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    st.write(f"Model (Random Forest) - Mean Absolute Error: {mae:.2f}")
    st.write(f"Model (Random Forest) - R² Score: {r2:.3f}")

    # 2. Forecasting with per-sector Linear Regression on Year
    selected_state = st.selectbox("Select State for Forecast", unique_states)

    max_year = max(years)
    future_year = st.slider(
        "Select Future Year",
        min_value=max_year + 1,
        max_value=max_year + 5,
        value=max_year + 1,
    )

    region = df_long.loc[df_long["State"] == selected_state, "Region"].mode()[0]

    prev_preds = []
    fut_preds = []

    for sector in unique_sectors:
        sub = df_long[
            (df_long["State"] == selected_state)
            & (df_long["Sector"] == sector)
        ].sort_values("Year")

        if sub["Year"].nunique() >= 2:
            # Fit linear regression: Year -> Employment
            X_sec = sub[["Year"]].values
            y_sec = sub["Employment"].values

            lr = LinearRegression()
            lr.fit(X_sec, y_sec)

            prev_val = lr.predict([[max_year]])[0]
            fut_val = lr.predict([[future_year]])[0]
        else:
            # Not enough data for trend; keep last known value
            if not sub.empty:
                last_val = sub["Employment"].iloc[-1]
            else:
                last_val = 0.0
            prev_val = last_val
            fut_val = last_val

        prev_preds.append(prev_val)
        fut_preds.append(fut_val)

    res = pd.DataFrame(
        {
            "Sector": unique_sectors,
            f"Employment_{max_year}": prev_preds,
            f"Employment_{future_year}": fut_preds,
        }
    )
    res["Growth"] = res[f"Employment_{future_year}"] - res[f"Employment_{max_year}"]

    # Fastest-growing sector
    fastest = res.loc[res["Growth"].idxmax()]

    st.subheader(f"Predicted Employment in {selected_state} for {future_year}")
    st.dataframe(res)

    st.markdown(
        f"Fastest growing sector in {selected_state} between {max_year} and {future_year}: "
        f"{fastest['Sector']} (growth = {fastest['Growth']:.2f})"
    )

    fig_pred = px.bar(
        res,
        x="Sector",
        y=f"Employment_{future_year}",
        title=f"Forecasted Employment in {selected_state} ({future_year})",
    )
    st.plotly_chart(fig_pred, use_container_width=True)
