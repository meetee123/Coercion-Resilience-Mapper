"""
Pivot Coercion Resilience Mapper
─────────────────────────────────
Commodity Corridor Exposure Simulator under Great-Power Rivalry

A multi-scenario resilience engine that maps small and middle powers'
strategic exposures in cocoa, gold, oil, and maritime corridors to coercion
by US/China/Russia/EU, then simulates state-level countermeasures accounting
for bounded rationality and path dependence.

Data sources:
  • UN Comtrade / WITS 2023
  • UNCTAD commodity statistics
  • World Bank WGI & FDI indicators
  • Ghana Ports & Harbours Authority
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import io
import json
from datetime import datetime

from data_engine import (
    COMMODITIES, GREAT_POWERS, COMMODITY_VALUES_M,
    COERCION_SHOCKS, COUNTERMEASURES, STATE_CAPACITY_INDEX,
    WGI_SCORES, PORT_THROUGHPUT, COMMODITY_SHARES, FDI_SHARES,
    build_baseline_snapshot, get_coercion_ladder_data,
    compute_exposure_matrix, compute_concentration_hhi,
)
from simulation_engine import (
    SimulationConfig, BoundedRationalityParams,
    run_monte_carlo, run_multi_commodity_comparison,
    rank_countermeasure_portfolios, compute_escalation_thresholds,
)

# ──────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Pivot Coercion Resilience Mapper",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────────────────────────
# CUSTOM STYLING
# ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    .stApp {
        font-family: 'Inter', sans-serif;
    }

    .main-header {
        font-size: 1.8rem;
        font-weight: 700;
        color: #0D2B2B;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }

    .sub-header {
        font-size: 1.0rem;
        font-weight: 400;
        color: #4A6E6E;
        margin-bottom: 1.5rem;
    }

    .metric-card {
        background: linear-gradient(135deg, #F0F7F7 0%, #E8F4F4 100%);
        border: 1px solid #C8DEDE;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }

    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #115058;
    }

    .metric-label {
        font-size: 0.75rem;
        font-weight: 500;
        color: #4A6E6E;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .risk-critical { color: #B91C1C; font-weight: 700; }
    .risk-high { color: #D97706; font-weight: 600; }
    .risk-moderate { color: #2563EB; font-weight: 500; }
    .risk-low { color: #059669; font-weight: 500; }

    div[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0D2B2B 0%, #13343B 100%);
    }

    div[data-testid="stSidebar"] .stMarkdown p,
    div[data-testid="stSidebar"] .stMarkdown li,
    div[data-testid="stSidebar"] label,
    div[data-testid="stSidebar"] .stSelectbox label,
    div[data-testid="stSidebar"] .stSlider label,
    div[data-testid="stSidebar"] .stMultiSelect label {
        color: #D6F5FA !important;
    }

    div[data-testid="stSidebar"] h1,
    div[data-testid="stSidebar"] h2,
    div[data-testid="stSidebar"] h3 {
        color: #FFFFFF !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-weight: 500;
    }

    .source-note {
        font-size: 0.7rem;
        color: #6B8A8A;
        margin-top: 0.5rem;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

# Color constants
BRAND_COLORS = {
    "teal": "#20808D",
    "terra": "#A84B2F",
    "dark_teal": "#1B474D",
    "cyan": "#BCE2E7",
    "mauve": "#944454",
    "gold": "#FFC553",
    "olive": "#848456",
    "brown": "#6E522B",
}

CHART_COLORS = list(BRAND_COLORS.values())

GP_COLORS = {
    "United States": "#2563EB",
    "China": "#DC2626",
    "Russia": "#7C3AED",
    "European Union": "#0D9488",
}

RISK_COLORS = {
    "Critical": "#B91C1C",
    "High": "#D97706",
    "Moderate": "#2563EB",
    "Low": "#059669",
}


# ──────────────────────────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌍 Coercion Resilience Mapper")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        [
            "📊 Executive Dashboard",
            "🔍 Exposure Analysis",
            "⚡ Shock Simulator",
            "🛡️ Countermeasure Lab",
            "🪜 Coercion Ladder",
            "📈 Portfolio Rankings",
            "📖 Methodology & Sources",
        ],
        index=0,
    )

    st.markdown("---")
    st.markdown("### Global Parameters")

    trade_w = st.slider("Trade weight", 0.0, 1.0, 0.45, 0.05, key="tw")
    fdi_w = st.slider("FDI weight", 0.0, 1.0, 0.30, 0.05, key="fw")
    logistics_w = round(1.0 - trade_w - fdi_w, 2)
    if logistics_w < 0:
        st.error("Weights must sum to ≤ 1.0")
        logistics_w = max(0, logistics_w)
    st.markdown(f"Logistics weight: **{logistics_w:.2f}**")

    state_cap = st.slider(
        "State capacity override",
        0.0, 1.0, STATE_CAPACITY_INDEX, 0.05,
        help="WGI-derived default ≈ 0.50; adjust to test higher/lower capacity scenarios."
    )

    n_mc = st.select_slider(
        "Monte Carlo runs",
        options=[50, 100, 200, 500, 1000],
        value=200,
        help="More runs = smoother distributions, slower compute."
    )

    st.markdown("---")
    st.markdown(
        '<p class="source-note">Data: UN Comtrade 2023, World Bank WGI, '
        'UNCTAD, GPHA Annual Reports</p>',
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────────────────────────

def render_metric_card(label, value, delta=None):
    """Render a styled metric card."""
    delta_html = ""
    if delta is not None:
        color = "#059669" if delta >= 0 else "#B91C1C"
        arrow = "▲" if delta >= 0 else "▼"
        delta_html = f'<div style="color:{color};font-size:0.85rem;font-weight:500;">{arrow} {abs(delta):.1f}%</div>'
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


def get_risk_class(level):
    return f"risk-{level.lower()}"


def dataframe_to_excel(df: pd.DataFrame) -> bytes:
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
        df.to_excel(writer, index=True, sheet_name='Data')
    return output.getvalue()


# ──────────────────────────────────────────────────────────────────
# PAGE: EXECUTIVE DASHBOARD
# ──────────────────────────────────────────────────────────────────

if page == "📊 Executive Dashboard":
    st.markdown('<div class="main-header">Pivot Coercion Resilience Mapper</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Commodity Corridor Exposure Simulator under Great-Power Rivalry</div>',
        unsafe_allow_html=True,
    )

    exposure_df, vuln_df = build_baseline_snapshot(trade_w, fdi_w, logistics_w)

    # Top metrics row
    col1, col2, col3, col4 = st.columns(4)
    total_export = sum(COMMODITY_VALUES_M.values())
    avg_vuln = vuln_df["Vulnerability Score"].mean()
    most_exposed_commodity = vuln_df["Vulnerability Score"].idxmax()
    most_exposed_power = vuln_df.loc[most_exposed_commodity, "Most Exposed To"]

    with col1:
        render_metric_card("Total Commodity Exports", f"${total_export/1000:.1f}B")
    with col2:
        render_metric_card("Avg Vulnerability Score", f"{avg_vuln:.3f}")
    with col3:
        render_metric_card("Most Exposed Sector", most_exposed_commodity)
    with col4:
        render_metric_card("Primary Coercion Risk", most_exposed_power)

    st.markdown("")

    # Two-column layout
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("#### Exposure Matrix: Commodity × Great Power")
        fig = px.imshow(
            exposure_df.values,
            x=exposure_df.columns.tolist(),
            y=exposure_df.index.tolist(),
            color_continuous_scale=[[0, "#F0F7F7"], [0.3, "#BCE2E7"], [0.6, "#20808D"], [1, "#0D2B2B"]],
            text_auto=".3f",
            aspect="auto",
        )
        fig.update_layout(
            height=320,
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Inter", size=12),
            coloraxis_colorbar=dict(title="Exposure", thickness=15),
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<p class="source-note">Source: Exposure index computed from UN Comtrade 2023 trade shares, '
            'World Bank FDI data, and UNCTAD maritime profiles. Weights: '
            f'Trade {trade_w:.0%}, FDI {fdi_w:.0%}, Logistics {logistics_w:.0%}.</p>',
            unsafe_allow_html=True,
        )

    with right:
        st.markdown("#### Sector Vulnerability Ranking")
        vuln_sorted = vuln_df.sort_values("Vulnerability Score", ascending=True)
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            y=vuln_sorted.index,
            x=vuln_sorted["Vulnerability Score"],
            orientation="h",
            marker_color=[CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(vuln_sorted))],
            text=[f'{v:.4f}' for v in vuln_sorted["Vulnerability Score"]],
            textposition="outside",
        ))
        fig2.update_layout(
            height=320,
            margin=dict(l=20, r=60, t=30, b=20),
            font=dict(family="Inter", size=12),
            xaxis_title="Vulnerability Score",
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(
            '<p class="source-note">Vulnerability = MaxExposure × HHI × (1 − StateCapacity) × GDPShareWeight. '
            'Higher = more susceptible to coercion.</p>',
            unsafe_allow_html=True,
        )

    # Trade composition sunburst
    st.markdown("#### Trade Dependency Composition by Commodity and Partner")
    sunburst_data = []
    for comm in COMMODITIES:
        for partner, share in COMMODITY_SHARES[comm].items():
            if share > 0:
                sunburst_data.append({
                    "Commodity": comm,
                    "Partner": partner,
                    "Share": share,
                    "Value ($M)": round(COMMODITY_VALUES_M[comm] * share, 1),
                })
    sb_df = pd.DataFrame(sunburst_data)
    fig3 = px.sunburst(
        sb_df,
        path=["Commodity", "Partner"],
        values="Value ($M)",
        color="Commodity",
        color_discrete_sequence=CHART_COLORS,
    )
    fig3.update_layout(
        height=450,
        margin=dict(l=20, r=20, t=30, b=20),
        font=dict(family="Inter", size=12),
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        '<p class="source-note">Source: UN Comtrade / WITS 2023. '
        'Values in USD millions. "Others" includes ASEAN, Gulf states, India, Japan, etc.</p>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────
# PAGE: EXPOSURE ANALYSIS
# ──────────────────────────────────────────────────────────────────

elif page == "🔍 Exposure Analysis":
    st.markdown('<div class="main-header">Exposure Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Sector-specific exposure indices from trade, FDI, and logistics data</div>',
        unsafe_allow_html=True,
    )

    exposure_df, vuln_df = build_baseline_snapshot(trade_w, fdi_w, logistics_w)

    tab1, tab2, tab3 = st.tabs(["📊 Exposure Matrix", "🏭 Sectoral Deep Dive", "🌐 Network View"])

    with tab1:
        st.markdown("##### Full Exposure Matrix")
        st.dataframe(
            exposure_df.style.background_gradient(cmap="YlGnBu", axis=None).format("{:.4f}"),
            use_container_width=True,
        )

        st.markdown("##### Vulnerability Assessment")
        st.dataframe(
            vuln_df.style.background_gradient(
                subset=["Vulnerability Score"], cmap="OrRd"
            ).format({
                "HHI": "{:.4f}",
                "GDP Share": "{:.2%}",
                "Max Single-Power Exposure": "{:.4f}",
                "Vulnerability Score": "{:.4f}",
            }),
            use_container_width=True,
        )

        col_dl1, col_dl2 = st.columns(2)
        with col_dl1:
            st.download_button(
                "📥 Export Exposure Matrix (CSV)",
                exposure_df.to_csv(),
                "exposure_matrix.csv",
                "text/csv",
            )
        with col_dl2:
            st.download_button(
                "📥 Export Vulnerability Table (Excel)",
                dataframe_to_excel(vuln_df),
                "vulnerability_assessment.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    with tab2:
        selected_commodity = st.selectbox("Select commodity sector", COMMODITIES)

        left2, right2 = st.columns(2)

        with left2:
            st.markdown(f"##### Trade Partners: {selected_commodity}")
            trade_data = pd.DataFrame([
                {"Partner": k, "Share": v}
                for k, v in COMMODITY_SHARES[selected_commodity].items()
                if v > 0
            ]).sort_values("Share", ascending=False)

            fig = px.bar(
                trade_data, x="Partner", y="Share",
                color="Partner",
                color_discrete_sequence=CHART_COLORS,
                text_auto=".1%",
            )
            fig.update_layout(
                height=350, showlegend=False,
                margin=dict(l=20, r=20, t=20, b=20),
                yaxis_tickformat=".0%",
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with right2:
            st.markdown(f"##### FDI Composition: {selected_commodity}")
            fdi_data = pd.DataFrame([
                {"Investor": k, "Share": v}
                for k, v in FDI_SHARES[selected_commodity].items()
                if v > 0
            ]).sort_values("Share", ascending=False)

            fig2 = px.pie(
                fdi_data, names="Investor", values="Share",
                color_discrete_sequence=CHART_COLORS,
                hole=0.4,
            )
            fig2.update_layout(
                height=350,
                margin=dict(l=20, r=20, t=20, b=20),
                font=dict(family="Inter"),
            )
            st.plotly_chart(fig2, use_container_width=True)

        # HHI and concentration
        hhi = compute_concentration_hhi(COMMODITY_SHARES[selected_commodity])
        c1, c2, c3 = st.columns(3)
        with c1:
            render_metric_card("HHI (Partner Concentration)", f"{hhi:.4f}")
        with c2:
            render_metric_card(
                "Export Value",
                f"${COMMODITY_VALUES_M[selected_commodity]:,.0f}M"
            )
        with c3:
            conc_level = "High" if hhi > 0.15 else "Moderate" if hhi > 0.10 else "Low"
            render_metric_card("Concentration Level", conc_level)

    with tab3:
        st.markdown("##### Commodity–Power Exposure Network")
        st.markdown(
            "Node size = export value; edge width = exposure strength. "
            "Thicker edges indicate higher coercion leverage."
        )

        import networkx as nx

        G = nx.Graph()
        for comm in COMMODITIES:
            G.add_node(comm, node_type="commodity", size=COMMODITY_VALUES_M[comm] / 500)
        for gp in GREAT_POWERS:
            G.add_node(gp, node_type="power", size=20)

        for comm in COMMODITIES:
            for gp in GREAT_POWERS:
                w = exposure_df.loc[comm, gp]
                if w > 0.01:
                    G.add_edge(comm, gp, weight=w)

        pos = nx.spring_layout(G, k=2.5, seed=42)

        edge_traces = []
        for edge in G.edges(data=True):
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            weight = edge[2]["weight"]
            edge_traces.append(go.Scatter(
                x=[x0, x1, None], y=[y0, y1, None],
                mode="lines",
                line=dict(width=weight * 15, color="rgba(32,128,141,0.4)"),
                hoverinfo="none",
                showlegend=False,
            ))

        node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
        for node in G.nodes(data=True):
            x, y = pos[node[0]]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node[0])
            if node[1]["node_type"] == "commodity":
                node_color.append(BRAND_COLORS["teal"])
                node_size.append(node[1]["size"])
            else:
                node_color.append(GP_COLORS.get(node[0], BRAND_COLORS["terra"]))
                node_size.append(node[1]["size"])

        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode="markers+text",
            text=node_text,
            textposition="top center",
            textfont=dict(size=11, family="Inter"),
            marker=dict(size=node_size, color=node_color, line=dict(width=1.5, color="white")),
            hoverinfo="text",
            showlegend=False,
        )

        fig_net = go.Figure(data=edge_traces + [node_trace])
        fig_net.update_layout(
            height=500,
            showlegend=False,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_net, use_container_width=True)

        st.markdown(
            '<p class="source-note">Network layout: spring-force algorithm. '
            'Edge thickness proportional to composite exposure index.</p>',
            unsafe_allow_html=True,
        )


# ──────────────────────────────────────────────────────────────────
# PAGE: SHOCK SIMULATOR
# ──────────────────────────────────────────────────────────────────

elif page == "⚡ Shock Simulator":
    st.markdown('<div class="main-header">Coercion Shock Simulator</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Monte Carlo simulation of great-power coercion shocks with uncertainty bands</div>',
        unsafe_allow_html=True,
    )

    col_left, col_right = st.columns([1, 1])
    with col_left:
        sim_commodity = st.selectbox("Target commodity", COMMODITIES, index=0)
        sim_power = st.selectbox("Coercing great power", GREAT_POWERS, index=3)
    with col_right:
        sim_shock = st.selectbox("Shock type", list(COERCION_SHOCKS.keys()), index=0)
        sim_horizon = st.slider("Time horizon (months)", 12, 120, 60, 6)

    # Countermeasure selection
    sim_cms = st.multiselect(
        "Active countermeasures",
        list(COUNTERMEASURES.keys()),
        default=["AfCFTA Rerouting", "Export Diversification"],
    )

    with st.expander("🧠 Bounded Rationality Parameters", expanded=False):
        br_col1, br_col2, br_col3 = st.columns(3)
        with br_col1:
            br_attention = st.slider("Attention bandwidth", 0.1, 1.0, 0.6, 0.05)
            br_loss_aversion = st.slider("Loss aversion (λ)", 1.0, 4.0, 2.25, 0.25)
        with br_col2:
            br_sq_bias = st.slider("Status-quo bias", 0.0, 0.8, 0.30, 0.05)
            br_satisfice = st.slider("Satisficing threshold", 0.3, 0.9, 0.60, 0.05)
        with br_col3:
            br_discount = st.slider("Discount rate", 0.01, 0.20, 0.08, 0.01)
            br_learning = st.slider("Learning rate", 0.01, 0.15, 0.05, 0.01)

    if st.button("▶ Run Simulation", type="primary", use_container_width=True):
        br_params = BoundedRationalityParams(
            attention_bandwidth=br_attention,
            loss_aversion_lambda=br_loss_aversion,
            status_quo_bias=br_sq_bias,
            satisficing_threshold=br_satisfice,
            discount_rate=br_discount,
            learning_rate=br_learning,
        )

        config = SimulationConfig(
            commodity=sim_commodity,
            coercing_power=sim_power,
            shock_type=sim_shock,
            time_horizon_months=sim_horizon,
            n_monte_carlo=n_mc,
            countermeasures_selected=sim_cms,
            state_capacity_override=state_cap,
            bounded_rationality=br_params,
        )

        with st.spinner(f"Running {n_mc} Monte Carlo paths..."):
            summary, all_paths, terminal = run_monte_carlo(config, seed=42)

        # Results
        st.markdown("---")

        # KPI row
        k1, k2, k3, k4 = st.columns(4)
        with k1:
            render_metric_card(
                "Mean Terminal Resilience",
                f"{terminal['Resilience Ratio'].mean():.1%}"
            )
        with k2:
            render_metric_card(
                "Worst Case (P5)",
                f"{terminal['Resilience Ratio'].quantile(0.05):.1%}"
            )
        with k3:
            render_metric_card(
                "Best Case (P95)",
                f"{terminal['Resilience Ratio'].quantile(0.95):.1%}"
            )
        with k4:
            render_metric_card(
                "Cumulative Loss",
                f"${summary['Mean_Loss'].sum():,.0f}M"
            )

        # Main chart: resilience fan
        st.markdown("#### Resilience Trajectory with Uncertainty Bands")
        fig = go.Figure()

        # P5-P95 band
        fig.add_trace(go.Scatter(
            x=pd.concat([summary["Month"], summary["Month"][::-1]]),
            y=pd.concat([summary["P95_Resilience"], summary["P5_Resilience"][::-1]]),
            fill="toself",
            fillcolor="rgba(32,128,141,0.12)",
            line=dict(color="rgba(0,0,0,0)"),
            name="P5–P95 range",
            showlegend=True,
        ))

        # P25-P75 band
        fig.add_trace(go.Scatter(
            x=pd.concat([summary["Month"], summary["Month"][::-1]]),
            y=pd.concat([summary["P75_Resilience"], summary["P25_Resilience"][::-1]]),
            fill="toself",
            fillcolor="rgba(32,128,141,0.25)",
            line=dict(color="rgba(0,0,0,0)"),
            name="P25–P75 range",
            showlegend=True,
        ))

        # Median line
        fig.add_trace(go.Scatter(
            x=summary["Month"],
            y=summary["P50_Resilience"],
            mode="lines",
            line=dict(color=BRAND_COLORS["teal"], width=3),
            name="Median (P50)",
        ))

        # Baseline reference
        fig.add_hline(y=1.0, line_dash="dash", line_color="#999", annotation_text="Baseline")

        # Critical threshold
        fig.add_hline(
            y=0.50, line_dash="dot", line_color="#B91C1C",
            annotation_text="Critical Threshold (50%)",
        )

        fig.update_layout(
            height=420,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Resilience Ratio (recovered / baseline)",
            xaxis_title="Months After Shock",
            yaxis_tickformat=".0%",
            font=dict(family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Decomposition chart
        st.markdown("#### Loss & Recovery Decomposition")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=summary["Month"], y=summary["Mean_Loss"],
            fill="tozeroy", name="Mean Loss ($M)",
            fillcolor="rgba(185,28,28,0.2)", line=dict(color="#B91C1C"),
        ))
        fig2.add_trace(go.Scatter(
            x=summary["Month"], y=summary["Mean_Recovery"],
            fill="tozeroy", name="Mean Recovery ($M)",
            fillcolor="rgba(5,150,105,0.2)", line=dict(color="#059669"),
        ))
        fig2.add_trace(go.Scatter(
            x=summary["Month"], y=summary["Mean_Residual_Gap"],
            mode="lines", name="Residual Gap ($M)",
            line=dict(color="#D97706", width=2, dash="dash"),
        ))
        fig2.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis_title="USD Millions",
            xaxis_title="Months After Shock",
            font=dict(family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Terminal distribution
        st.markdown("#### Terminal Resilience Distribution")
        fig3 = px.histogram(
            terminal, x="Resilience Ratio", nbins=40,
            color_discrete_sequence=[BRAND_COLORS["teal"]],
            marginal="box",
        )
        fig3.add_vline(
            x=terminal["Resilience Ratio"].mean(),
            line_dash="dash", line_color=BRAND_COLORS["terra"],
            annotation_text=f'Mean: {terminal["Resilience Ratio"].mean():.3f}',
        )
        fig3.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Terminal Resilience Ratio",
            xaxis_tickformat=".0%",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Export
        st.download_button(
            "📥 Export Simulation Results (CSV)",
            summary.to_csv(index=False),
            f"simulation_{sim_commodity}_{sim_power}_{sim_shock}.csv",
            "text/csv",
        )


# ──────────────────────────────────────────────────────────────────
# PAGE: COUNTERMEASURE LAB
# ──────────────────────────────────────────────────────────────────

elif page == "🛡️ Countermeasure Lab":
    st.markdown('<div class="main-header">Countermeasure Laboratory</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Explore adaptive response mechanisms with path-dependence dynamics</div>',
        unsafe_allow_html=True,
    )

    # Countermeasure catalog
    st.markdown("#### Countermeasure Catalog")
    cm_records = []
    for name, info in COUNTERMEASURES.items():
        cm_records.append({
            "Name": name,
            "Trade Recovery %": f"{info['trade_recovery_pct']:.0%}",
            "Time to Effect": f"{info['time_to_effect_months']} mo",
            "Cost (% GDP)": f"{info['cost_gdp_pct']:.1%}",
            "Capacity Threshold": f"{info['state_capacity_threshold']:.0%}",
            "Path Dependence": f"{info['path_dependence_factor']:.1f}",
            "Description": info["description"],
        })
    cm_df = pd.DataFrame(cm_records)
    st.dataframe(cm_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # Comparative simulation
    st.markdown("#### Comparative Countermeasure Simulation")

    cm_col1, cm_col2 = st.columns(2)
    with cm_col1:
        cm_commodity = st.selectbox("Commodity", COMMODITIES, key="cm_comm")
        cm_power = st.selectbox("Coercing power", GREAT_POWERS, key="cm_pow", index=3)
    with cm_col2:
        cm_shock = st.selectbox("Shock type", list(COERCION_SHOCKS.keys()), key="cm_shock")

    # Side-by-side A/B comparison
    st.markdown("##### Compare Two Countermeasure Portfolios")
    ab_col1, ab_col2 = st.columns(2)
    with ab_col1:
        st.markdown("**Portfolio A**")
        portfolio_a = st.multiselect(
            "Select measures (A)",
            list(COUNTERMEASURES.keys()),
            default=["AfCFTA Rerouting", "Export Diversification"],
            key="port_a",
        )
    with ab_col2:
        st.markdown("**Portfolio B**")
        portfolio_b = st.multiselect(
            "Select measures (B)",
            list(COUNTERMEASURES.keys()),
            default=["Value-Chain Upgrading", "WTO Complaint / Legal Signaling"],
            key="port_b",
        )

    if st.button("▶ Compare Portfolios", type="primary", use_container_width=True):
        with st.spinner("Running A/B comparison..."):
            # Portfolio A
            cfg_a = SimulationConfig(
                commodity=cm_commodity, coercing_power=cm_power,
                shock_type=cm_shock, countermeasures_selected=portfolio_a,
                n_monte_carlo=n_mc, time_horizon_months=60,
                state_capacity_override=state_cap,
            )
            sum_a, _, term_a = run_monte_carlo(cfg_a, seed=42)

            # Portfolio B
            cfg_b = SimulationConfig(
                commodity=cm_commodity, coercing_power=cm_power,
                shock_type=cm_shock, countermeasures_selected=portfolio_b,
                n_monte_carlo=n_mc, time_horizon_months=60,
                state_capacity_override=state_cap,
            )
            sum_b, _, term_b = run_monte_carlo(cfg_b, seed=42)

        # Comparison chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=sum_a["Month"], y=sum_a["P50_Resilience"],
            mode="lines", line=dict(color=BRAND_COLORS["teal"], width=3),
            name="Portfolio A (Median)",
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([sum_a["Month"], sum_a["Month"][::-1]]),
            y=pd.concat([sum_a["P75_Resilience"], sum_a["P25_Resilience"][::-1]]),
            fill="toself", fillcolor="rgba(32,128,141,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="Portfolio A (IQR)",
        ))
        fig.add_trace(go.Scatter(
            x=sum_b["Month"], y=sum_b["P50_Resilience"],
            mode="lines", line=dict(color=BRAND_COLORS["terra"], width=3),
            name="Portfolio B (Median)",
        ))
        fig.add_trace(go.Scatter(
            x=pd.concat([sum_b["Month"], sum_b["Month"][::-1]]),
            y=pd.concat([sum_b["P75_Resilience"], sum_b["P25_Resilience"][::-1]]),
            fill="toself", fillcolor="rgba(168,75,47,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="Portfolio B (IQR)",
        ))
        fig.add_hline(y=1.0, line_dash="dash", line_color="#999")

        fig.update_layout(
            height=420,
            yaxis_title="Resilience Ratio",
            xaxis_title="Months After Shock",
            yaxis_tickformat=".0%",
            font=dict(family="Inter"),
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Summary comparison table
        comp_df = pd.DataFrame({
            "Metric": [
                "Mean Terminal Resilience",
                "Worst Case (P5)",
                "Best Case (P95)",
                "Total Cost (% GDP)",
            ],
            "Portfolio A": [
                f"{term_a['Resilience Ratio'].mean():.1%}",
                f"{term_a['Resilience Ratio'].quantile(0.05):.1%}",
                f"{term_a['Resilience Ratio'].quantile(0.95):.1%}",
                f"{sum(COUNTERMEASURES[c]['cost_gdp_pct'] for c in portfolio_a):.2%}",
            ],
            "Portfolio B": [
                f"{term_b['Resilience Ratio'].mean():.1%}",
                f"{term_b['Resilience Ratio'].quantile(0.05):.1%}",
                f"{term_b['Resilience Ratio'].quantile(0.95):.1%}",
                f"{sum(COUNTERMEASURES[c]['cost_gdp_pct'] for c in portfolio_b):.2%}",
            ],
        })
        st.dataframe(comp_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────
# PAGE: COERCION LADDER
# ──────────────────────────────────────────────────────────────────

elif page == "🪜 Coercion Ladder":
    st.markdown('<div class="main-header">Coercion Ladder Heatmap</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Escalation thresholds: how resilient is each sector to each shock type, by great power?</div>',
        unsafe_allow_html=True,
    )

    # Coercion shock catalog
    with st.expander("📋 Coercion Shock Catalog", expanded=False):
        ladder_df = get_coercion_ladder_data()
        st.dataframe(ladder_df, use_container_width=True, hide_index=True)

    cl_commodity = st.selectbox("Commodity sector", COMMODITIES, key="cl_comm")
    cl_cms = st.multiselect(
        "Active countermeasures",
        list(COUNTERMEASURES.keys()),
        default=["AfCFTA Rerouting", "Export Diversification"],
        key="cl_cms",
    )

    if st.button("▶ Generate Coercion Ladder", type="primary", use_container_width=True):
        with st.spinner("Computing escalation thresholds across all great powers and shock types..."):
            esc_df = compute_escalation_thresholds(
                commodity=cl_commodity,
                countermeasures=cl_cms,
                n_mc=min(n_mc, 150),
                seed=42,
            )

        # Pivot for heatmap
        pivot = esc_df.pivot(
            index="Shock Type", columns="Great Power",
            values="Mean Terminal Resilience",
        )

        # Order shocks by average severity
        shock_order = pivot.mean(axis=1).sort_values().index.tolist()
        pivot = pivot.loc[shock_order]

        st.markdown(f"#### Escalation Heatmap: {cl_commodity}")
        fig = px.imshow(
            pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            color_continuous_scale=[
                [0.0, "#67000D"], [0.3, "#D32F2F"],
                [0.5, "#FF8F00"], [0.7, "#FFC553"],
                [0.85, "#BCE2E7"], [1.0, "#E8F5E9"],
            ],
            text_auto=".1%",
            aspect="auto",
            zmin=0.3, zmax=1.0,
        )
        fig.update_layout(
            height=450,
            margin=dict(l=20, r=20, t=30, b=20),
            font=dict(family="Inter", size=12),
            coloraxis_colorbar=dict(title="Resilience", tickformat=".0%"),
            xaxis_title="Coercing Great Power",
            yaxis_title="Escalation Level (most severe → least severe)",
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(
            '<p class="source-note">Cells show mean terminal resilience ratio after 48 months. '
            'Red = critical vulnerability; green = robust resilience. '
            'Rows ordered by average severity across all powers.</p>',
            unsafe_allow_html=True,
        )

        # Risk table
        st.markdown("#### Detailed Risk Assessment")
        esc_display = esc_df.copy()
        esc_display["Mean Terminal Resilience"] = esc_display["Mean Terminal Resilience"].apply(
            lambda x: f"{x:.1%}"
        )
        st.dataframe(
            esc_display,
            use_container_width=True,
            hide_index=True,
        )

        st.download_button(
            "📥 Export Coercion Ladder (CSV)",
            esc_df.to_csv(index=False),
            f"coercion_ladder_{cl_commodity}.csv",
            "text/csv",
        )


# ──────────────────────────────────────────────────────────────────
# PAGE: PORTFOLIO RANKINGS
# ──────────────────────────────────────────────────────────────────

elif page == "📈 Portfolio Rankings":
    st.markdown('<div class="main-header">Resilience Portfolio Rankings</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Ranked countermeasure portfolios with cost-adjusted scoring</div>',
        unsafe_allow_html=True,
    )

    pr_col1, pr_col2, pr_col3 = st.columns(3)
    with pr_col1:
        pr_commodity = st.selectbox("Commodity", COMMODITIES, key="pr_comm")
    with pr_col2:
        pr_power = st.selectbox("Coercing power", GREAT_POWERS, key="pr_pow", index=3)
    with pr_col3:
        pr_shock = st.selectbox("Shock type", list(COERCION_SHOCKS.keys()), key="pr_shock")

    if st.button("▶ Rank Portfolios", type="primary", use_container_width=True):
        with st.spinner("Evaluating all countermeasure portfolios..."):
            rankings = rank_countermeasure_portfolios(
                commodity=pr_commodity,
                coercing_power=pr_power,
                shock_type=pr_shock,
                n_mc=n_mc,
                time_horizon=60,
                seed=42,
            )

        st.markdown("#### Portfolio Performance Ranking")

        # Bar chart of rankings
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=rankings["Portfolio"],
            x=rankings["Mean Terminal Resilience"],
            orientation="h",
            marker_color=[CHART_COLORS[i % len(CHART_COLORS)] for i in range(len(rankings))],
            text=[f'{v:.1%}' for v in rankings["Mean Terminal Resilience"]],
            textposition="outside",
            name="Mean Resilience",
        ))

        # Error bars (P5 to P95)
        fig.add_trace(go.Scatter(
            y=rankings["Portfolio"],
            x=rankings["P5 (Worst Case)"],
            mode="markers",
            marker=dict(symbol="line-ew", size=12, color="#B91C1C", line_width=2),
            name="P5 (Worst)",
        ))
        fig.add_trace(go.Scatter(
            y=rankings["Portfolio"],
            x=rankings["P95 (Best Case)"],
            mode="markers",
            marker=dict(symbol="line-ew", size=12, color="#059669", line_width=2),
            name="P95 (Best)",
        ))

        fig.update_layout(
            height=500,
            margin=dict(l=20, r=80, t=30, b=20),
            xaxis_title="Terminal Resilience Ratio",
            xaxis_tickformat=".0%",
            font=dict(family="Inter"),
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            barmode="overlay",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Full table
        st.markdown("#### Detailed Rankings")
        display_rankings = rankings.copy()
        for col in ["Mean Terminal Resilience", "P5 (Worst Case)", "P95 (Best Case)"]:
            display_rankings[col] = display_rankings[col].apply(lambda x: f"{x:.1%}")
        display_rankings["Estimated Cost (% GDP)"] = display_rankings["Estimated Cost (% GDP)"].apply(
            lambda x: f"{x:.2f}%"
        )
        st.dataframe(display_rankings, use_container_width=True, hide_index=True)

        # Scatter: resilience vs cost
        st.markdown("#### Efficiency Frontier: Resilience vs. Cost")
        fig2 = px.scatter(
            rankings,
            x="Estimated Cost (% GDP)",
            y="Mean Terminal Resilience",
            text="Portfolio",
            color="Mean Terminal Resilience",
            color_continuous_scale=[[0, "#B91C1C"], [0.5, "#FFC553"], [1, "#059669"]],
            size="Cost-Adjusted Score",
            size_max=30,
        )
        fig2.update_traces(textposition="top center", textfont_size=10)
        fig2.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Mean Terminal Resilience",
            yaxis_tickformat=".0%",
            xaxis_title="Estimated Cost (% GDP)",
            font=dict(family="Inter"),
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button(
            "📥 Export Rankings (Excel)",
            dataframe_to_excel(rankings),
            f"portfolio_rankings_{pr_commodity}.xlsx",
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )

    # Multi-commodity comparison section
    st.markdown("---")
    st.markdown("#### Cross-Commodity Comparison")
    st.markdown("Compare resilience across all commodity sectors under the same shock scenario.")

    if st.button("▶ Run Cross-Commodity Analysis", use_container_width=True):
        with st.spinner("Running cross-commodity analysis..."):
            cross_df = run_multi_commodity_comparison(
                coercing_power=pr_power,
                shock_type=pr_shock,
                n_mc=min(n_mc, 200),
                seed=42,
            )

        fig3 = go.Figure()
        for i, row in cross_df.iterrows():
            fig3.add_trace(go.Bar(
                x=[row["Commodity"]],
                y=[row["Mean Terminal Resilience"]],
                name=row["Commodity"],
                marker_color=CHART_COLORS[i % len(CHART_COLORS)],
                text=f'{row["Mean Terminal Resilience"]:.1%}',
                textposition="outside",
                error_y=dict(
                    type="data",
                    symmetric=False,
                    array=[row["P95 Terminal Resilience"] - row["Mean Terminal Resilience"]],
                    arrayminus=[row["Mean Terminal Resilience"] - row["P5 Terminal Resilience"]],
                ),
            ))

        fig3.update_layout(
            height=400,
            yaxis_title="Terminal Resilience Ratio",
            yaxis_tickformat=".0%",
            showlegend=False,
            font=dict(family="Inter"),
            margin=dict(l=20, r=20, t=30, b=20),
        )
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(cross_df, use_container_width=True, hide_index=True)


# ──────────────────────────────────────────────────────────────────
# PAGE: METHODOLOGY & SOURCES
# ──────────────────────────────────────────────────────────────────

elif page == "📖 Methodology & Sources":
    st.markdown('<div class="main-header">Methodology & Sources</div>', unsafe_allow_html=True)

    st.markdown("""
    ### Theoretical Framework

    This tool integrates three distinct analytical traditions:

    **1. Commodity Weaponisation Literature**
    - Farrell & Newman (2019): "Weaponized Interdependence" — how network centrality enables coercion
    - Drezner (2021): sanctions efficacy and asymmetric power in commodity markets
    - Bown (2022): export controls and supply-chain coercion in the US–China context
    - SIPRI (2026): Resource mercantilism and great-power perils for the global rest

    **2. Bounded Rationality & Decision Theory**
    - Kahneman & Tversky (1979): Prospect theory — loss aversion (λ ≈ 2.25) in policy decisions
    - Simon (1956): Satisficing under uncertainty — states accept "good enough" outcomes
    - March (1991): Exploration vs. exploitation in organisational adaptation
    - Path dependence: lock-in effects from prior strategic alignments (Arthur, 1989)

    **3. Small-State Strategic Studies**
    - Keohane (1969): "Lilliputian's Dilemmas" — constraints facing small states
    - Thorhallsson (2012): small-state shelter theory
    - AfCFTA trade diversion potential (UNCTAD Economic Development in Africa Report 2024)

    ---

    ### Data Sources & Calibration

    | Data Domain | Source | Year | Access |
    |---|---|---|---|
    | Trade flows (cocoa, gold, oil) | UN Comtrade via WITS | 2023 | [WITS](https://wits.worldbank.org/) |
    | FDI stocks by country | World Bank / UNCTAD | 2023 | [World Bank](https://data.worldbank.org/indicator/BX.KLT.DINV.CD.WD) |
    | Governance indicators | World Bank WGI | 2023 | [WGI](https://www.worldbank.org/en/publication/worldwide-governance-indicators) |
    | Port throughput | Ghana Ports & Harbours Authority | 2023–24 | [GPHA Reports](https://ghanaports.gov.gh/) |
    | Maritime shipping profiles | UNCTAD Data Hub | 2023 | [UNCTAD](https://unctadstat.unctad.org/) |
    | Gold export data | Ghana Gold Board (GoldBod) | 2023 | [GoldBod](https://goldbod.gov.gh/) |
    | Shock precedents | Brookings, UNCTAD, SIPRI | 2022–26 | Various |

    ---

    ### Exposure Index Methodology

    The composite exposure index for commodity *i* vis-à-vis great power *j* is:

    **E_ij = w₁ · TradeShare_ij + w₂ · FDIShare_ij + w₃ · LogisticsProxy_ij**

    Where:
    - **w₁** (Trade weight): Share of commodity exports directed to the great power
    - **w₂** (FDI weight): Share of sector-specific FDI originating from the great power
    - **w₃** (Logistics weight): Port/shipping dependency on the great power's logistics networks

    Default weights: Trade 45%, FDI 30%, Logistics 25% (adjustable via sidebar).

    ---

    ### Vulnerability Scoring

    **V_i = max(E_ij) × HHI_i × (1 − StateCapacity) × (0.5 + GDPShare_i)**

    Where:
    - **HHI_i**: Herfindahl–Hirschman Index of partner concentration (excluding "Others")
    - **StateCapacity**: Normalised WGI composite (default ≈ 0.50 for Ghana)
    - **GDPShare_i**: Sector's share of total commodity exports

    ---

    ### Simulation Engine

    **Monte Carlo Architecture:**
    - *N* independent simulation paths (default 200; adjustable 50–1000)
    - Each path: monthly time-step over the shock horizon
    - Stochastic noise (σ = 0.08) injected at each step

    **Shock Dynamics:**
    - 3-month ramp-up phase
    - Plateau at peak disruption for the shock's defined duration
    - Exponential decay (λ = 0.10/month) after duration expires

    **Countermeasure Model:**
    - Logistic activation curve centred on each measure's time-to-effect
    - Bounded-rationality filters: attention bandwidth, loss aversion, status-quo bias
    - Path-dependence penalties: higher lock-in → greater status-quo friction
    - State-capacity multiplier: weak institutions reduce countermeasure efficacy
    - Bayesian learning: incremental improvement over time (default 5%/period)
    - Diminishing returns on combined countermeasures

    ---

    ### Uncertainty & Limitations

    **Explicit uncertainty quantification:**
    - All results presented with P5/P25/P50/P75/P95 bands
    - Terminal resilience shown as full distribution, not point estimate

    **Known limitations:**
    - Trade-share data is 2023-vintage; real-time shifts not captured
    - FDI breakdown by sector is approximate (country-level data disaggregated by sector proxies)
    - Maritime corridor dependency is a composite proxy, not direct measurement
    - Model assumes shock parameters are independent across commodities (real-world correlation may differ)
    - Bounded-rationality parameters are drawn from experimental economics literature, not calibrated to specific country decision-making processes
    - "Others" category in trade data may mask significant bilateral relationships

    **Mitigation strategies:**
    - Stochastic noise injection prevents false precision
    - Sensitivity analysis via adjustable weights and BR parameters
    - Open methodology allows expert override of all assumptions

    ---

    ### Citation

    If using this tool in publications or policy documents, please cite:

    > Pivot Coercion Resilience Mapper: Commodity Corridor Exposure Simulator
    > under Great-Power Rivalry. Built with Streamlit. Data: UN Comtrade 2023,
    > World Bank WGI, UNCTAD, GPHA. Methodology draws on Farrell & Newman (2019),
    > Kahneman & Tversky (1979), and UNCTAD EDAR 2024.
    """)

    # State capacity detail
    st.markdown("---")
    st.markdown("### Ghana Governance Profile (WGI 2023)")
    wgi_df = pd.DataFrame([
        {"Indicator": k, "Percentile Rank": v}
        for k, v in WGI_SCORES.items()
    ])
    fig_wgi = px.bar(
        wgi_df, x="Indicator", y="Percentile Rank",
        color="Percentile Rank",
        color_continuous_scale=[[0, "#B91C1C"], [0.5, "#FFC553"], [1, "#059669"]],
        text_auto=".1f",
    )
    fig_wgi.update_layout(
        height=350,
        yaxis_range=[0, 100],
        margin=dict(l=20, r=20, t=20, b=20),
        font=dict(family="Inter"),
        showlegend=False,
    )
    st.plotly_chart(fig_wgi, use_container_width=True)

    st.markdown(f"""
    **Composite State Capacity Index:** {STATE_CAPACITY_INDEX:.3f}
    (mean of six WGI percentile ranks, normalised 0–1)

    **Port Infrastructure:**
    - Tema Port: {PORT_THROUGHPUT['Tema']:.1f} million tonnes (2023)
    - Takoradi Port: {PORT_THROUGHPUT['Takoradi']:.1f} million tonnes (2023)
    - Total: {sum(PORT_THROUGHPUT.values()):.1f} million tonnes

    Source: [Ghana Ports & Harbours Authority](https://ghanaports.gov.gh/),
    [World Bank WGI](https://www.worldbank.org/en/publication/worldwide-governance-indicators)
    """)
