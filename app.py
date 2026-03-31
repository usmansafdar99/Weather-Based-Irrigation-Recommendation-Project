from __future__ import annotations

from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import DATA_PATH
from src.data import load_dataset
from src.irrigation_logic import classify_water_requirement
from src.modeling import load_model, train_and_save_default_model
from src.preprocessing import engineer_features


@st.cache_resource
def get_model():
    try:
        return load_model()
    except FileNotFoundError:
        # Streamlit Community Cloud: model artifact is often not committed.
        # Train once and cache the trained pipeline in-memory.
        with st.spinner("Training model for the first time (one-time setup)..."):
            return train_and_save_default_model()


@st.cache_data
def get_reference_data():
    return load_dataset(DATA_PATH)


def inject_custom_css() -> None:
    st.markdown(
        """
<style>
/* App background – cleaner, softer gradient */
.stApp {
  background: radial-gradient(1200px 800px at 0% 0%, rgba(59,130,246,0.18), transparent 60%),
              radial-gradient(1100px 900px at 100% 0%, rgba(16,185,129,0.18), transparent 55%),
              linear-gradient(180deg, #020617 0%, #020617 45%, #020617 100%);
  color: #E5E7EB;
  font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}

/* Container spacing */
div.block-container { padding-top: 0.7rem; max-width: 1180px; }

/* Headings */
h1, h2, h3 { color: #F9FAFB; }

/* Sidebar */
section[data-testid="stSidebar"] {
  background: rgba(2, 6, 23, 0.88);
  border-right: 1px solid rgba(148,163,184,0.18);
}

/* Card */
.card {
  background: linear-gradient(135deg, rgba(15,23,42,0.94), rgba(15,23,42,0.86));
  border: 1px solid rgba(148,163,184,0.18);
  border-radius: 18px;
  padding: 18px 20px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.25);
}
.muted { color: rgba(226,232,240,0.75); }
.pill {
  display: inline-block;
  padding: 6px 10px;
  border-radius: 999px;
  font-weight: 600;
  letter-spacing: 0.2px;
  border: 1px solid rgba(148,163,184,0.25);
  background: rgba(2,6,23,0.45);
}

/* Button */
div.stButton > button {
  background: linear-gradient(90deg, rgba(56,189,248,1) 0%, rgba(34,197,94,1) 100%);
  border: 0;
  color: #0B1020;
  font-weight: 700;
  border-radius: 12px;
  padding: 0.65rem 1.1rem;
}
div.stButton > button:hover {
  filter: brightness(1.05);
}

/* Dataframe */
div[data-testid="stDataFrame"] {
  border-radius: 14px;
  overflow: hidden;
  border: 1px solid rgba(148,163,184,0.18);
}
</style>
        """,
        unsafe_allow_html=True,
    )


def build_input_form(df_example: pd.DataFrame) -> pd.DataFrame:
    st.sidebar.markdown("### Farm & Weather Inputs")
    st.sidebar.caption("Adjust conditions to get a daily + weekly irrigation recommendation.")

    soil_type = st.sidebar.selectbox("Soil Type", sorted(df_example["Soil_Type"].unique()))
    soil_pH = st.sidebar.slider(
        "Soil pH",
        float(df_example["Soil_pH"].min()),
        float(df_example["Soil_pH"].max()),
        float(df_example["Soil_pH"].median()),
    )
    soil_moisture = st.sidebar.slider(
        "Soil Moisture",
        float(df_example["Soil_Moisture"].min()),
        float(df_example["Soil_Moisture"].max()),
        float(df_example["Soil_Moisture"].median()),
    )
    organic_carbon = st.sidebar.slider(
        "Organic Carbon",
        float(df_example["Organic_Carbon"].min()),
        float(df_example["Organic_Carbon"].max()),
        float(df_example["Organic_Carbon"].median()),
    )
    ec = st.sidebar.slider(
        "Electrical Conductivity",
        float(df_example["Electrical_Conductivity"].min()),
        float(df_example["Electrical_Conductivity"].max()),
        float(df_example["Electrical_Conductivity"].median()),
    )
    temp_c = st.sidebar.slider(
        "Temperature (°C)",
        float(df_example["Temperature_C"].min()),
        float(df_example["Temperature_C"].max()),
        float(df_example["Temperature_C"].median()),
    )
    humidity = st.sidebar.slider(
        "Humidity (%)",
        float(df_example["Humidity"].min()),
        float(df_example["Humidity"].max()),
        float(df_example["Humidity"].median()),
    )
    rainfall = st.sidebar.slider(
        "Rainfall (mm)",
        float(df_example["Rainfall_mm"].min()),
        float(df_example["Rainfall_mm"].max()),
        float(df_example["Rainfall_mm"].median()),
    )
    sunlight = st.sidebar.slider(
        "Sunlight Hours",
        float(df_example["Sunlight_Hours"].min()),
        float(df_example["Sunlight_Hours"].max()),
        float(df_example["Sunlight_Hours"].median()),
    )
    wind_speed = st.sidebar.slider(
        "Wind Speed (km/h)",
        float(df_example["Wind_Speed_kmh"].min()),
        float(df_example["Wind_Speed_kmh"].max()),
        float(df_example["Wind_Speed_kmh"].median()),
    )

    crop_type = st.sidebar.selectbox("Crop Type", sorted(df_example["Crop_Type"].unique()))
    crop_stage = st.sidebar.selectbox("Crop Growth Stage", sorted(df_example["Crop_Growth_Stage"].unique()))
    season = st.sidebar.selectbox("Season", sorted(df_example["Season"].unique()))
    irrigation_type = st.sidebar.selectbox("Irrigation Type", sorted(df_example["Irrigation_Type"].unique()))
    water_source = st.sidebar.selectbox("Water Source", sorted(df_example["Water_Source"].unique()))

    field_area = st.sidebar.slider(
        "Field Area (hectare)",
        float(df_example["Field_Area_hectare"].min()),
        float(df_example["Field_Area_hectare"].max()),
        float(df_example["Field_Area_hectare"].median()),
    )

    mulching_used = st.sidebar.selectbox("Mulching Used", sorted(df_example["Mulching_Used"].unique()))
    previous_irrigation = st.sidebar.slider(
        "Previous Irrigation (mm)",
        float(df_example["Previous_Irrigation_mm"].min()),
        float(df_example["Previous_Irrigation_mm"].max()),
        float(df_example["Previous_Irrigation_mm"].median()),
    )
    region = st.sidebar.selectbox("Region", sorted(df_example["Region"].unique()))

    input_dict = {
        "Soil_Type": soil_type,
        "Soil_pH": soil_pH,
        "Soil_Moisture": soil_moisture,
        "Organic_Carbon": organic_carbon,
        "Electrical_Conductivity": ec,
        "Temperature_C": temp_c,
        "Humidity": humidity,
        "Rainfall_mm": rainfall,
        "Sunlight_Hours": sunlight,
        "Wind_Speed_kmh": wind_speed,
        "Crop_Type": crop_type,
        "Crop_Growth_Stage": crop_stage,
        "Season": season,
        "Irrigation_Type": irrigation_type,
        "Water_Source": water_source,
        "Field_Area_hectare": field_area,
        "Mulching_Used": mulching_used,
        "Previous_Irrigation_mm": previous_irrigation,
        "Region": region,
    }

    return pd.DataFrame([input_dict])


def init_session_state() -> None:
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def push_history_row(input_df: pd.DataFrame, pred_daily: float, level: str) -> None:
    row = input_df.iloc[0].to_dict()
    row.update(
        {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "pred_daily_score": float(pred_daily),
            "pred_weekly_score": float(pred_daily) * 7.0,
            "pred_level": level,
        }
    )
    st.session_state.prediction_history.append(row)


def recommendation_gauge(pred_daily: float) -> go.Figure:
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(pred_daily),
            number={"font": {"size": 40, "color": "#F8FAFC"}},
            title={"text": "Daily water requirement (score ~ mm)", "font": {"color": "#E2E8F0"}},
            gauge={
                "axis": {"range": [0, 15], "tickcolor": "#94A3B8"},
                "bar": {"color": "#22C55E"},
                "bgcolor": "rgba(2,6,23,0.0)",
                "borderwidth": 1,
                "bordercolor": "rgba(148,163,184,0.25)",
                "steps": [
                    {"range": [0, 3], "color": "rgba(34,197,94,0.25)"},
                    {"range": [3, 8], "color": "rgba(234,179,8,0.28)"},
                    {"range": [8, 15], "color": "rgba(239,68,68,0.28)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=60, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#E5E7EB"),
    )
    return fig


def main():
    st.set_page_config(page_title="Weather-Based Irrigation Recommendation", layout="wide")
    inject_custom_css()
    init_session_state()

    st.markdown(
        """
<div class="card" style="margin-bottom: 0.75rem;">
  <div style="display:flex; justify-content:space-between; align-items:flex-start; gap:14px; flex-wrap:wrap;">
    <div>
      <div class="pill">Weather • Soil • Crop</div>
      <h1 style="margin:10px 0 0 0;">Smart Irrigation Advisor</h1>
      <p class="muted" style="margin:6px 0 0 0; font-size: 14px;">
        Use your local weather and soil conditions to predict crop water needs and get clear irrigation actions.
      </p>
    </div>
    <div style="text-align:right;">
      <div class="muted" style="font-size:12px;">Model-driven recommendation</div>
      <div style="font-weight:700; font-size:14px;">Local • Offline-ready</div>
    </div>
  </div>
</div>
        """,
        unsafe_allow_html=True,
    )

    model = get_model()
    df_example = get_reference_data()

    tabs = st.tabs(["💧 Predict", "📈 Prediction History", "📊 Dataset Insights"])

    with tabs[0]:
        input_df = build_input_form(df_example)

        # Apply same feature engineering as training
        input_df_engineered = engineer_features(input_df)

        left, right = st.columns([1.1, 0.9], gap="large")

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Irrigation prediction")
            st.caption("Run the model with your current settings from the left sidebar.")
            run = st.button("Predict irrigation recommendation")

            if run:
                pred_daily = float(model.predict(input_df_engineered)[0])
                rec = classify_water_requirement(pred_daily)
                push_history_row(input_df, pred_daily, rec.level)

                st.markdown("<hr style='border-color: rgba(30,64,175,0.4);' />", unsafe_allow_html=True)
                k1, k2, k3 = st.columns(3)
                k1.metric("Daily water need", f"{rec.water_requirement_daily:.2f}")
                k2.metric("Weekly water need", f"{rec.water_requirement_weekly:.2f}")
                k3.metric("Irrigation level", rec.level)

                st.markdown(
                    f"""
<div class="card" style="border-left: 6px solid {rec.color}; margin-top: 10px;">
  <div class="pill">Recommendation</div>
  <h3 style="margin:10px 0 0 0; color:{rec.color};"> {rec.level} </h3>
  <p class="muted" style="margin:6px 0 0 0;">{rec.message}</p>
</div>
                    """,
                    unsafe_allow_html=True,
                )

                st.markdown('<div class="card" style="margin-top:10px;">', unsafe_allow_html=True)
                st.subheader("Input snapshot")
                st.dataframe(input_df, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Visual summary")
            st.caption("Gauge and bar chart update with your latest prediction.")

            if len(st.session_state.prediction_history) > 0:
                last = st.session_state.prediction_history[-1]
                pred_daily = float(last["pred_daily_score"])
                pred_weekly = float(last["pred_weekly_score"])

                fig_g = recommendation_gauge(pred_daily)
                st.plotly_chart(fig_g, use_container_width=True)

                fig_week = px.bar(
                    pd.DataFrame({"Period": ["Daily", "Weekly"], "Score": [pred_daily, pred_weekly]}),
                    x="Period",
                    y="Score",
                    color="Period",
                    color_discrete_map={"Daily": "#38BDF8", "Weekly": "#22C55E"},
                    title="Daily vs weekly requirement (latest prediction)",
                )
                fig_week.update_layout(
                    height=320,
                    margin=dict(l=10, r=10, t=50, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E5E7EB"),
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                )
                fig_week.update_xaxes(showgrid=False)
                fig_week.update_yaxes(gridcolor="rgba(148,163,184,0.18)")
                st.plotly_chart(fig_week, use_container_width=True)
            else:
                st.info("Run a prediction to see the gauge and bar chart here.")

    with tabs[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prediction History")
        st.caption("This tracks your predictions in the current session (so you can compare scenarios).")
        st.markdown("</div>", unsafe_allow_html=True)

        history = st.session_state.prediction_history
        if len(history) == 0:
            st.info("No predictions yet. Go to the **Predict** tab and run your first prediction.")
        else:
            hist_df = pd.DataFrame(history)
            c1, c2 = st.columns([0.62, 0.38], gap="large")

            with c1:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Daily requirement trend")
                trend = px.line(
                    hist_df,
                    x="timestamp",
                    y="pred_daily_score",
                    markers=True,
                    title="Predicted daily water requirement over time",
                )
                trend.update_layout(
                    height=360,
                    margin=dict(l=10, r=10, t=50, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E5E7EB"),
                )
                trend.update_xaxes(showgrid=False)
                trend.update_yaxes(gridcolor="rgba(148,163,184,0.18)")
                st.plotly_chart(trend, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            with c2:
                st.markdown('<div class="card">', unsafe_allow_html=True)
                st.subheader("Level distribution")
                dist = px.histogram(
                    hist_df,
                    x="pred_level",
                    color="pred_level",
                    category_orders={"pred_level": ["Low", "Medium", "High"]},
                    color_discrete_map={"Low": "#22C55E", "Medium": "#EAB308", "High": "#EF4444"},
                    title="How often each recommendation occurs",
                )
                dist.update_layout(
                    height=360,
                    margin=dict(l=10, r=10, t=50, b=10),
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    font=dict(color="#E5E7EB"),
                    showlegend=False,
                )
                dist.update_xaxes(showgrid=False)
                dist.update_yaxes(gridcolor="rgba(148,163,184,0.18)")
                st.plotly_chart(dist, use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
            st.subheader("History table")
            st.dataframe(
                hist_df[
                    [
                        "timestamp",
                        "Crop_Type",
                        "Season",
                        "Region",
                        "Soil_Moisture",
                        "Rainfall_mm",
                        "Temperature_C",
                        "Humidity",
                        "pred_daily_score",
                        "pred_level",
                    ]
                ],
                use_container_width=True,
            )
            b1, b2 = st.columns([0.2, 0.8])
            with b1:
                if st.button("Clear history"):
                    st.session_state.prediction_history = []
                    st.rerun()
            st.markdown("</div>", unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Dataset Insights (interactive)")
        st.caption("Quick visual exploration using your dataset (helps validate patterns).")
        st.markdown("</div>", unsafe_allow_html=True)

        d = df_example.copy()
        # Interactive scatter: weather vs soil moisture
        left, right = st.columns(2, gap="large")

        with left:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Rainfall vs Soil Moisture")
            fig1 = px.scatter(
                d,
                x="Rainfall_mm",
                y="Soil_Moisture",
                color="Irrigation_Need",
                hover_data=["Crop_Type", "Season", "Region"],
                opacity=0.65,
                title="Rainfall vs soil moisture (colored by Irrigation_Need)",
            )
            fig1.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E5E7EB"),
                legend=dict(bgcolor="rgba(2,6,23,0.35)"),
            )
            fig1.update_xaxes(gridcolor="rgba(148,163,184,0.18)")
            fig1.update_yaxes(gridcolor="rgba(148,163,184,0.18)")
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with right:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Temperature vs Soil Moisture")
            fig2 = px.scatter(
                d,
                x="Temperature_C",
                y="Soil_Moisture",
                color="Irrigation_Need",
                hover_data=["Crop_Type", "Season", "Region"],
                opacity=0.65,
                title="Temperature vs soil moisture (colored by Irrigation_Need)",
            )
            fig2.update_layout(
                height=420,
                margin=dict(l=10, r=10, t=50, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#E5E7EB"),
                legend=dict(bgcolor="rgba(2,6,23,0.35)"),
            )
            fig2.update_xaxes(gridcolor="rgba(148,163,184,0.18)")
            fig2.update_yaxes(gridcolor="rgba(148,163,184,0.18)")
            st.plotly_chart(fig2, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown('<div class="card" style="margin-top:12px;">', unsafe_allow_html=True)
        st.subheader("Irrigation need breakdown")
        fig3 = px.pie(
            d,
            names="Irrigation_Need",
            title="Class distribution in dataset",
            color="Irrigation_Need",
            color_discrete_map={"Low": "#22C55E", "Medium": "#EAB308", "High": "#EF4444"},
        )
        fig3.update_layout(
            height=360,
            margin=dict(l=10, r=10, t=50, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#E5E7EB"),
            legend=dict(bgcolor="rgba(2,6,23,0.35)"),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()

