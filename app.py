"""
Pivot Coercion Resilience Mapper
─────────────────────────────────
Commodity Corridor Exposure Simulator under Great-Power Rivalry

A multi-scenario resilience engine that maps small and middle powers'
strategic exposures in cocoa, gold, oil, and maritime corridors to coercion
by US/China/Russia/EU, then simulates state-level countermeasures accounting
for bounded rationality and path dependence.

Data: UN Comtrade / WITS 2023, World Bank WGI, UNCTAD, GPHA.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import io
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

# ═══════════════════════════════════════════════════════════════════
# SECTION 1 ─ DATA ENGINE
# All calibrated baseline data, exposure indices, shock & countermeasure
# definitions.  Sources cited inline.
# ═══════════════════════════════════════════════════════════════════

COMMODITIES = ["Cocoa", "Gold", "Oil", "Maritime Corridor"]
GREAT_POWERS = ["United States", "China", "Russia", "European Union"]

# ── Trade-flow shares (UN Comtrade / WITS 2023) ────────────────────
# Ghana cocoa exports 2023: ~$1.1 B.  NL $264 M, US $145 M, MY $140 M,
# BE $131 M, JP $86 M, FR $42 M, DE $42 M, ES $61 M, IT $37 M, etc.
COCOA_SHARES = {"European Union": 0.52, "United States": 0.13,
                "China": 0.01, "Russia": 0.00, "Others": 0.34}
# Ghana gold exports 2023: ~$7.6 B semi-manufactured.
# CH $3.1 B, UAE $1.7 B, ZA $1.7 B, IN $950 M, TR $244 M
GOLD_SHARES = {"European Union": 0.41, "United States": 0.02,
               "China": 0.01, "Russia": 0.00, "Others": 0.56}
# Ghana oil: Jubilee & TEN fields.  CN ~25 %, IN ~15 %, EU ~20 %, US ~5 %
OIL_SHARES = {"European Union": 0.20, "United States": 0.05,
              "China": 0.25, "Russia": 0.00, "Others": 0.50}
# Maritime corridor proxy: port investment, shipping lines, naval presence
# Tema + Takoradi 26 M t (2023) – Maersk/MSC/CMA-CGM (EU), COSCO (CN)
MARITIME_SHARES = {"European Union": 0.40, "United States": 0.10,
                   "China": 0.30, "Russia": 0.05, "Others": 0.15}

COMMODITY_SHARES = {"Cocoa": COCOA_SHARES, "Gold": GOLD_SHARES,
                    "Oil": OIL_SHARES, "Maritime Corridor": MARITIME_SHARES}

# Absolute export values (USD millions, 2023)
COMMODITY_VALUES_M = {"Cocoa": 1_107, "Gold": 7_632,
                      "Oil": 4_200, "Maritime Corridor": 2_600}

# ── FDI stock shares (World Bank / UNCTAD) ─────────────────────────
FDI_SHARES = {
    "Cocoa":    {"European Union": 0.35, "United States": 0.15,
                 "China": 0.05, "Russia": 0.00, "Others": 0.45},
    "Gold":     {"European Union": 0.25, "United States": 0.10,
                 "China": 0.20, "Russia": 0.02, "Others": 0.43},
    "Oil":      {"European Union": 0.30, "United States": 0.20,
                 "China": 0.25, "Russia": 0.00, "Others": 0.25},
    "Maritime Corridor": {"European Union": 0.35, "United States": 0.10,
                          "China": 0.35, "Russia": 0.05, "Others": 0.15},
}

# ── Governance / state-capacity (WGI 2023) ─────────────────────────
WGI_SCORES = {
    "Voice & Accountability": 62.6,
    "Political Stability": 46.2,
    "Government Effectiveness": 43.8,
    "Regulatory Quality": 50.5,
    "Rule of Law": 53.3,
    "Control of Corruption": 41.0,
}
STATE_CAPACITY_INDEX = np.mean(list(WGI_SCORES.values())) / 100.0

PORT_THROUGHPUT = {"Tema": 18.0, "Takoradi": 8.0}

# ── Coercion shock catalogue ──────────────────────────────────────
COERCION_SHOCKS = {
    "Export Ban (full)": {
        "description": "Complete ban on imports of the commodity from Ghana.",
        "trade_disruption": 1.0, "fdi_flight_pct": 0.60,
        "price_impact_pct": -0.30, "duration_months": 12,
        "precedent": "Russian grain export restrictions 2022; US sanctions on Iranian oil",
    },
    "Price Cap / Buyer Cartel": {
        "description": "Coordinated price ceiling imposed by major buyers.",
        "trade_disruption": 0.40, "fdi_flight_pct": 0.20,
        "price_impact_pct": -0.25, "duration_months": 18,
        "precedent": "G7 Russian oil price cap ($60/bbl) Dec 2022",
    },
    "Port Pressure / Logistics Denial": {
        "description": "Denial of port services, insurance, or shipping access.",
        "trade_disruption": 0.70, "fdi_flight_pct": 0.30,
        "price_impact_pct": -0.15, "duration_months": 6,
        "precedent": "Houthi Red Sea shipping disruption 2023-2024",
    },
    "Sanctions on Key Entities": {
        "description": "Targeted sanctions on state trading companies or mining firms.",
        "trade_disruption": 0.50, "fdi_flight_pct": 0.40,
        "price_impact_pct": -0.10, "duration_months": 24,
        "precedent": "US OFAC sanctions on Venezuelan PDVSA; EU sanctions on Russian gold",
    },
    "FDI Weaponisation": {
        "description": "Withdrawal or freezing of FDI as political leverage.",
        "trade_disruption": 0.20, "fdi_flight_pct": 0.80,
        "price_impact_pct": -0.05, "duration_months": 36,
        "precedent": "China investment freezes in Australia 2020-2021",
    },
    "Diplomatic Pressure (Soft Coercion)": {
        "description": "Implicit threats tied to aid, debt restructuring, or votes.",
        "trade_disruption": 0.10, "fdi_flight_pct": 0.10,
        "price_impact_pct": -0.05, "duration_months": 6,
        "precedent": "Chinese debt-trap diplomacy debates; US AGOA conditionality",
    },
}

# ── Countermeasure catalogue ──────────────────────────────────────
COUNTERMEASURES = {
    "AfCFTA Rerouting": {
        "description": "Redirect exports through AfCFTA partner corridors.",
        "trade_recovery_pct": 0.40, "time_to_effect_months": 12,
        "cost_gdp_pct": 0.02, "state_capacity_threshold": 0.40,
        "path_dependence_factor": 0.3,
    },
    "Export Diversification": {
        "description": "Cultivate alternative buyer markets (ASEAN, Gulf, India).",
        "trade_recovery_pct": 0.55, "time_to_effect_months": 18,
        "cost_gdp_pct": 0.03, "state_capacity_threshold": 0.45,
        "path_dependence_factor": 0.5,
    },
    "Value-Chain Upgrading": {
        "description": "Process raw commodities domestically (cocoa->chocolate, gold refining).",
        "trade_recovery_pct": 0.30, "time_to_effect_months": 36,
        "cost_gdp_pct": 0.05, "state_capacity_threshold": 0.50,
        "path_dependence_factor": 0.7,
    },
    "WTO Complaint / Legal Signaling": {
        "description": "File WTO dispute or invoke bilateral investment treaty arbitration.",
        "trade_recovery_pct": 0.15, "time_to_effect_months": 24,
        "cost_gdp_pct": 0.005, "state_capacity_threshold": 0.35,
        "path_dependence_factor": 0.1,
    },
    "Strategic Stockpiling": {
        "description": "Build buffer stocks to ride out temporary supply disruptions.",
        "trade_recovery_pct": 0.20, "time_to_effect_months": 6,
        "cost_gdp_pct": 0.04, "state_capacity_threshold": 0.30,
        "path_dependence_factor": 0.2,
    },
    "Counter-Alignment (Pivot)": {
        "description": "Shift strategic alignment toward a rival great power for protection.",
        "trade_recovery_pct": 0.60, "time_to_effect_months": 6,
        "cost_gdp_pct": 0.01, "state_capacity_threshold": 0.25,
        "path_dependence_factor": 0.8,
    },
    "Hedging via Commodity Futures": {
        "description": "Use financial instruments to lock in prices and reduce volatility.",
        "trade_recovery_pct": 0.10, "time_to_effect_months": 3,
        "cost_gdp_pct": 0.01, "state_capacity_threshold": 0.45,
        "path_dependence_factor": 0.1,
    },
}


# ── Exposure index computation ────────────────────────────────────

def compute_exposure_matrix(tw=0.45, fw=0.30, lw=0.25):
    """Sector x Great-Power composite exposure.
    E_ij = w_trade*TradeShare + w_fdi*FDIShare + w_logistics*LogisticsProxy"""
    rows = []
    for c in COMMODITIES:
        row = {}
        for gp in GREAT_POWERS:
            ts = COMMODITY_SHARES[c].get(gp, 0)
            fs = FDI_SHARES[c].get(gp, 0)
            ls = MARITIME_SHARES.get(gp, 0) if c == "Maritime Corridor" else ts * 0.8
            row[gp] = round(tw * ts + fw * fs + lw * ls, 4)
        rows.append(row)
    return pd.DataFrame(rows, index=COMMODITIES)


def compute_hhi(shares):
    """Herfindahl-Hirschman Index (excluding 'Others')."""
    return sum(v ** 2 for k, v in shares.items() if k != "Others")


def compute_vulnerability(exposure_df):
    """V_i = max(E_ij) * HHI * (1 - StateCapacity) * (0.5 + GDPShare)"""
    total = sum(COMMODITY_VALUES_M.values())
    recs = []
    for c in COMMODITIES:
        hhi = compute_hhi(COMMODITY_SHARES[c])
        gdp_s = COMMODITY_VALUES_M[c] / total
        max_e = exposure_df.loc[c].max()
        vuln = max_e * hhi * (1 - STATE_CAPACITY_INDEX) * (0.5 + gdp_s)
        recs.append({"Commodity": c, "HHI": round(hhi, 4),
                      "GDP Share": round(gdp_s, 4),
                      "Max Exposure": round(max_e, 4),
                      "Most Exposed To": exposure_df.loc[c].idxmax(),
                      "Vulnerability": round(vuln, 4)})
    return pd.DataFrame(recs).set_index("Commodity")


def get_coercion_ladder():
    """Shocks ordered by composite escalation severity."""
    recs = []
    for name, s in COERCION_SHOCKS.items():
        esc = round(0.4 * s["trade_disruption"] + 0.3 * s["fdi_flight_pct"]
                    + 0.2 * abs(s["price_impact_pct"])
                    + 0.1 * min(s["duration_months"] / 36, 1), 3)
        recs.append({"Shock Type": name, "Trade Disruption": s["trade_disruption"],
                      "FDI Flight %": s["fdi_flight_pct"],
                      "Price Impact %": s["price_impact_pct"],
                      "Duration (mo)": s["duration_months"],
                      "Description": s["description"],
                      "Historical Precedent": s["precedent"],
                      "Escalation Score": esc})
    return pd.DataFrame(recs).sort_values("Escalation Score").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
# SECTION 2 ─ SIMULATION ENGINE
# Monte Carlo with bounded rationality + path dependence
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BRParams:
    """Bounded-rationality parameters (Kahneman-Tversky / Simon / March)."""
    attention: float = 0.6
    loss_aversion: float = 2.25
    sq_bias: float = 0.30
    satisfice: float = 0.60
    discount: float = 0.08
    learning: float = 0.05


@dataclass
class SimConfig:
    commodity: str = "Cocoa"
    power: str = "European Union"
    shock: str = "Export Ban (full)"
    horizon: int = 60
    n_mc: int = 200
    noise_std: float = 0.08
    state_cap: Optional[float] = None
    cms: List[str] = field(default_factory=lambda: [
        "AfCFTA Rerouting", "Export Diversification"])
    br: BRParams = field(default_factory=BRParams)


def _apply_shock(baseline, shock_p, exposure, month, noise):
    """Shock with ramp-up, plateau, decay, cascading amplifier, FDI drag."""
    dur = shock_p["duration_months"]
    amp = 1.5 + 0.5 * shock_p["trade_disruption"]
    peak = min(0.95, shock_p["trade_disruption"] * exposure * amp)
    price_eff = shock_p["price_impact_pct"]

    if month <= 3:
        ramp = month / 3.0
    elif month <= dur:
        ramp = 1.0
    else:
        ramp = np.exp(-0.06 * (month - dur))

    disr = peak * ramp
    pa = 1.0 + price_eff * ramp
    fdi_drag = shock_p["fdi_flight_pct"] * exposure * 0.15 * ramp
    nd = max(0, min(0.95, disr + fdi_drag + noise))
    return max(0, baseline * (1 - nd) * pa)


def _apply_cms(cur, base, cms_list, month, sc, br, rng):
    """Countermeasures filtered through bounded-rationality gates."""
    gap = base - cur
    if gap <= 0:
        return cur, {}
    total_rec = 0.0
    contribs = {}
    for cn in cms_list:
        cm = COUNTERMEASURES[cn]
        if sc < cm["state_capacity_threshold"]:
            contribs[cn] = 0.0
            continue
        act = 1.0 / (1.0 + np.exp(-0.3 * (month - cm["time_to_effect_months"])))
        att = 1.0 if rng.random() < br.attention else 0.4
        sq = 1.0 - br.sq_bias * cm["path_dependence_factor"]
        cp = max(0.3, 1.0 - cm["cost_gdp_pct"] * br.loss_aversion * 5)
        eff = (cm["trade_recovery_pct"] * act * att * sq * cp
               * (0.5 + 0.5 * sc)
               * (1.0 + br.learning * min(month, 36))
               * (1.0 + rng.normal(0, 0.05)))
        eff = max(0, min(0.95, eff))
        marg = gap * eff * max(0.3, 1.0 - total_rec / gap) if gap > 0 else 0
        total_rec += marg
        contribs[cn] = marg
    return cur + min(total_rec, gap), contribs


def run_single(cfg, rng):
    base = COMMODITY_VALUES_M[cfg.commodity]
    shock_p = COERCION_SHOCKS[cfg.shock]
    sc = cfg.state_cap if cfg.state_cap is not None else STATE_CAPACITY_INDEX
    exp_mat = compute_exposure_matrix()
    exp = exp_mat.loc[cfg.commodity, cfg.power]
    recs = []
    for m in range(1, cfg.horizon + 1):
        noise = rng.normal(0, cfg.noise_std)
        sv = _apply_shock(base, shock_p, exp, m, noise)
        rv, cc = _apply_cms(sv, base, cfg.cms, m, sc, cfg.br, rng)
        recs.append({"Month": m, "Baseline": base,
                      "Shocked": round(sv, 2), "Recovered": round(rv, 2),
                      "Loss": round(base - sv, 2),
                      "Recovery": round(rv - sv, 2),
                      "Residual Gap": round(base - rv, 2),
                      "Resilience": round(rv / base, 4) if base > 0 else 0})
    return pd.DataFrame(recs)


def run_mc(cfg, seed=42):
    """Returns (summary_per_month, terminal_distribution)."""
    rng = np.random.default_rng(seed)
    runs = []
    for i in range(cfg.n_mc):
        df = run_single(cfg, rng)
        df["Run"] = i
        runs.append(df)
    all_p = pd.concat(runs, ignore_index=True)

    summary = all_p.groupby("Month").agg(
        Mean=("Resilience", "mean"),
        P5=("Resilience", lambda x: np.percentile(x, 5)),
        P25=("Resilience", lambda x: np.percentile(x, 25)),
        P50=("Resilience", "median"),
        P75=("Resilience", lambda x: np.percentile(x, 75)),
        P95=("Resilience", lambda x: np.percentile(x, 95)),
        Mean_Loss=("Loss", "mean"),
        Mean_Rec=("Recovery", "mean"),
        Mean_Gap=("Residual Gap", "mean"),
        Mean_Shocked=("Shocked", "mean"),
        Mean_Recovered=("Recovered", "mean"),
    ).reset_index()

    term = all_p[all_p["Month"] == cfg.horizon][["Run", "Resilience", "Residual Gap", "Recovered"]].copy()
    return summary, term


def run_cross_commodity(power, shock, cms=None, n_mc=200, horizon=60, seed=42):
    if cms is None:
        cms = ["AfCFTA Rerouting", "Export Diversification"]
    recs = []
    for c in COMMODITIES:
        cfg = SimConfig(commodity=c, power=power, shock=shock, cms=cms,
                        n_mc=n_mc, horizon=horizon)
        s, t = run_mc(cfg, seed)
        recs.append({"Commodity": c,
                      "Baseline ($M)": COMMODITY_VALUES_M[c],
                      "Mean Terminal": round(t["Resilience"].mean(), 4),
                      "P5": round(t["Resilience"].quantile(0.05), 4),
                      "P95": round(t["Resilience"].quantile(0.95), 4),
                      "Cum. Loss ($M)": round(s["Mean_Loss"].sum(), 1),
                      "Cum. Recovery ($M)": round(s["Mean_Rec"].sum(), 1)})
    return pd.DataFrame(recs)


def rank_portfolios(commodity, power, shock, n_mc=200, horizon=60, seed=42):
    portfolios = {n: [n] for n in COUNTERMEASURES}
    portfolios["AfCFTA + Diversification"] = ["AfCFTA Rerouting", "Export Diversification"]
    portfolios["Full Defensive"] = ["AfCFTA Rerouting", "Export Diversification",
                                    "Strategic Stockpiling", "Hedging via Commodity Futures"]
    portfolios["Legal + Diplomatic"] = ["WTO Complaint / Legal Signaling",
                                        "Counter-Alignment (Pivot)"]
    portfolios["Industrial Policy"] = ["Value-Chain Upgrading", "AfCFTA Rerouting"]
    portfolios["No Response"] = []
    recs = []
    for pn, cl in portfolios.items():
        cfg = SimConfig(commodity=commodity, power=power, shock=shock,
                        cms=cl, n_mc=n_mc, horizon=horizon)
        _, t = run_mc(cfg, seed)
        tc = sum(COUNTERMEASURES[c]["cost_gdp_pct"] for c in cl)
        recs.append({"Portfolio": pn,
                      "Countermeasures": ", ".join(cl) if cl else "(none)",
                      "Mean Terminal": round(t["Resilience"].mean(), 4),
                      "P5 (Worst)": round(t["Resilience"].quantile(0.05), 4),
                      "P95 (Best)": round(t["Resilience"].quantile(0.95), 4),
                      "Cost (% GDP)": round(tc * 100, 2),
                      "Cost-Adj Score": round(t["Resilience"].mean() / max(tc * 100 + 0.1, 0.1), 4)})
    return pd.DataFrame(recs).sort_values("Mean Terminal", ascending=False).reset_index(drop=True)


def compute_escalation(commodity, cms=None, n_mc=150, seed=42):
    if cms is None:
        cms = ["AfCFTA Rerouting", "Export Diversification"]
    recs = []
    for gp in GREAT_POWERS:
        for sn in COERCION_SHOCKS:
            cfg = SimConfig(commodity=commodity, power=gp, shock=sn,
                            cms=cms, n_mc=n_mc, horizon=48)
            _, t = run_mc(cfg, seed)
            mr = t["Resilience"].mean()
            recs.append({"Great Power": gp, "Shock Type": sn,
                          "Mean Terminal Resilience": round(mr, 4),
                          "Risk": ("Critical" if mr < 0.50
                                   else "High" if mr < 0.70
                                   else "Moderate" if mr < 0.85
                                   else "Low")})
    return pd.DataFrame(recs)


# ═══════════════════════════════════════════════════════════════════
# SECTION 3 ─ STREAMLIT UI
# ═══════════════════════════════════════════════════════════════════

st.set_page_config(page_title="Pivot Coercion Resilience Mapper",
                   page_icon="🌍", layout="wide",
                   initial_sidebar_state="expanded")

# ── Styles ─────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
.stApp { font-family: 'Inter', sans-serif; }
.main-hdr { font-size:1.8rem; font-weight:700; color:#0D2B2B;
            margin-bottom:0.2rem; letter-spacing:-0.02em; }
.sub-hdr  { font-size:1.0rem; font-weight:400; color:#4A6E6E;
            margin-bottom:1.5rem; }
.mc { background:linear-gradient(135deg,#F0F7F7,#E8F4F4);
      border:1px solid #C8DEDE; border-radius:10px;
      padding:1.2rem; text-align:center; }
.mv { font-size:1.8rem; font-weight:700; color:#115058; }
.ml { font-size:0.75rem; font-weight:500; color:#4A6E6E;
      text-transform:uppercase; letter-spacing:0.05em; }
.sn { font-size:0.7rem; color:#6B8A8A; margin-top:0.5rem;
      font-style:italic; }
div[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#0D2B2B,#13343B); }
div[data-testid="stSidebar"] p,
div[data-testid="stSidebar"] li,
div[data-testid="stSidebar"] label { color:#D6F5FA !important; }
div[data-testid="stSidebar"] h1,
div[data-testid="stSidebar"] h2,
div[data-testid="stSidebar"] h3 { color:#FFF !important; }
</style>""", unsafe_allow_html=True)

COLORS = ["#20808D", "#A84B2F", "#1B474D", "#BCE2E7",
          "#944454", "#FFC553", "#848456", "#6E522B"]
GP_COL = {"United States": "#2563EB", "China": "#DC2626",
          "Russia": "#7C3AED", "European Union": "#0D9488"}


def mcard(label, value):
    st.markdown(f'<div class="mc"><div class="ml">{label}</div>'
                f'<div class="mv">{value}</div></div>',
                unsafe_allow_html=True)


def to_excel(df):
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as w:
        df.to_excel(w, index=True, sheet_name="Data")
    return buf.getvalue()


# ── Sidebar ────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## 🌍 Coercion Resilience Mapper")
    st.markdown("---")
    page = st.radio("Navigate", [
        "📊 Executive Dashboard",
        "🔍 Exposure Analysis",
        "⚡ Shock Simulator",
        "🛡️ Countermeasure Lab",
        "🪜 Coercion Ladder",
        "📈 Portfolio Rankings",
        "📖 Methodology & Sources",
    ], index=0)
    st.markdown("---")
    st.markdown("### Global Parameters")
    tw = st.slider("Trade weight", 0.0, 1.0, 0.45, 0.05)
    fw = st.slider("FDI weight", 0.0, 1.0, 0.30, 0.05)
    lw = round(1.0 - tw - fw, 2)
    if lw < 0:
        st.error("Weights must sum to <= 1.0"); lw = max(0, lw)
    st.markdown(f"Logistics weight: **{lw:.2f}**")
    sc_override = st.slider("State capacity override", 0.0, 1.0,
                            float(round(STATE_CAPACITY_INDEX, 2)), 0.05,
                            help="WGI default ~0.50")
    n_mc = st.select_slider("Monte Carlo runs",
                            options=[50, 100, 200, 500, 1000], value=200)
    st.markdown("---")
    st.markdown('<p class="sn">Data: UN Comtrade 2023, World Bank WGI, '
                'UNCTAD, GPHA</p>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: EXECUTIVE DASHBOARD
# ════════════════════════════════════════════════════════════════════

if page == "📊 Executive Dashboard":
    st.markdown('<div class="main-hdr">Pivot Coercion Resilience Mapper</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-hdr">Commodity Corridor Exposure Simulator '
                'under Great-Power Rivalry</div>', unsafe_allow_html=True)

    exp_df = compute_exposure_matrix(tw, fw, lw)
    vul_df = compute_vulnerability(exp_df)

    c1, c2, c3, c4 = st.columns(4)
    with c1: mcard("Total Exports", f"${sum(COMMODITY_VALUES_M.values())/1000:.1f}B")
    with c2: mcard("Avg Vulnerability", f"{vul_df['Vulnerability'].mean():.3f}")
    with c3: mcard("Most Exposed", vul_df["Vulnerability"].idxmax())
    with c4: mcard("Primary Risk", vul_df.loc[vul_df["Vulnerability"].idxmax(),
                                               "Most Exposed To"])

    st.markdown("")
    left, right = st.columns([1.1, 0.9])

    with left:
        st.markdown("#### Exposure Matrix: Commodity x Great Power")
        fig = px.imshow(exp_df.values, x=exp_df.columns.tolist(),
                        y=exp_df.index.tolist(),
                        color_continuous_scale=[[0,"#F0F7F7"],[0.3,"#BCE2E7"],
                                                [0.6,"#20808D"],[1,"#0D2B2B"]],
                        text_auto=".3f", aspect="auto")
        fig.update_layout(height=320, margin=dict(l=20,r=20,t=30,b=20),
                          font=dict(family="Inter",size=12),
                          coloraxis_colorbar=dict(title="Exposure",thickness=15))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<p class="sn">Source: UN Comtrade 2023 trade shares, '
                    'World Bank FDI, UNCTAD maritime profiles. '
                    f'Weights: Trade {tw:.0%}, FDI {fw:.0%}, '
                    f'Logistics {lw:.0%}.</p>', unsafe_allow_html=True)

    with right:
        st.markdown("#### Sector Vulnerability Ranking")
        vs = vul_df.sort_values("Vulnerability", ascending=True)
        fig2 = go.Figure(go.Bar(
            y=vs.index, x=vs["Vulnerability"], orientation="h",
            marker_color=[COLORS[i % len(COLORS)] for i in range(len(vs))],
            text=[f'{v:.4f}' for v in vs["Vulnerability"]],
            textposition="outside"))
        fig2.update_layout(height=320, margin=dict(l=20,r=60,t=30,b=20),
                           font=dict(family="Inter",size=12),
                           xaxis_title="Vulnerability Score", showlegend=False)
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown('<p class="sn">V = MaxExposure x HHI x (1-StateCapacity) '
                    'x GDPShareWeight</p>', unsafe_allow_html=True)

    # Sunburst
    st.markdown("#### Trade Dependency Composition")
    sb = []
    for co in COMMODITIES:
        for p, s in COMMODITY_SHARES[co].items():
            if s > 0:
                sb.append({"Commodity": co, "Partner": p,
                           "Value ($M)": round(COMMODITY_VALUES_M[co] * s, 1)})
    fig3 = px.sunburst(pd.DataFrame(sb), path=["Commodity", "Partner"],
                       values="Value ($M)", color="Commodity",
                       color_discrete_sequence=COLORS)
    fig3.update_layout(height=450, margin=dict(l=20,r=20,t=30,b=20),
                       font=dict(family="Inter",size=12))
    st.plotly_chart(fig3, use_container_width=True)
    st.markdown('<p class="sn">Source: UN Comtrade/WITS 2023. '
                'Values in USD millions.</p>', unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: EXPOSURE ANALYSIS
# ════════════════════════════════════════════════════════════════════

elif page == "🔍 Exposure Analysis":
    st.markdown('<div class="main-hdr">Exposure Analysis</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-hdr">Sector-specific exposure indices from '
                'trade, FDI, and logistics data</div>', unsafe_allow_html=True)

    exp_df = compute_exposure_matrix(tw, fw, lw)
    vul_df = compute_vulnerability(exp_df)

    tab1, tab2 = st.tabs(["📊 Exposure Matrix", "🏭 Sectoral Deep Dive"])

    with tab1:
        st.markdown("##### Full Exposure Matrix")
        st.dataframe(exp_df.style.background_gradient(cmap="YlGnBu", axis=None)
                     .format("{:.4f}"), use_container_width=True)
        st.markdown("##### Vulnerability Assessment")
        st.dataframe(vul_df.style.background_gradient(
            subset=["Vulnerability"], cmap="OrRd").format({
                "HHI":"{:.4f}", "GDP Share":"{:.2%}",
                "Max Exposure":"{:.4f}", "Vulnerability":"{:.4f}"}),
            use_container_width=True)
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Exposure CSV", exp_df.to_csv(),
                               "exposure_matrix.csv", "text/csv")
        with c2:
            st.download_button("📥 Vulnerability Excel", to_excel(vul_df),
                               "vulnerability.xlsx",
                               "application/vnd.openxmlformats-officedocument"
                               ".spreadsheetml.sheet")

    with tab2:
        sel = st.selectbox("Select commodity", COMMODITIES)
        l2, r2 = st.columns(2)
        with l2:
            st.markdown(f"##### Trade Partners: {sel}")
            td = pd.DataFrame([{"Partner": k, "Share": v}
                               for k, v in COMMODITY_SHARES[sel].items()
                               if v > 0]).sort_values("Share", ascending=False)
            fig = px.bar(td, x="Partner", y="Share", color="Partner",
                         color_discrete_sequence=COLORS, text_auto=".1%")
            fig.update_layout(height=350, showlegend=False,
                              margin=dict(l=20,r=20,t=20,b=20),
                              yaxis_tickformat=".0%",
                              font=dict(family="Inter"))
            st.plotly_chart(fig, use_container_width=True)
        with r2:
            st.markdown(f"##### FDI Composition: {sel}")
            fd = pd.DataFrame([{"Investor": k, "Share": v}
                               for k, v in FDI_SHARES[sel].items()
                               if v > 0]).sort_values("Share", ascending=False)
            fig2 = px.pie(fd, names="Investor", values="Share",
                          color_discrete_sequence=COLORS, hole=0.4)
            fig2.update_layout(height=350, margin=dict(l=20,r=20,t=20,b=20),
                               font=dict(family="Inter"))
            st.plotly_chart(fig2, use_container_width=True)

        hhi = compute_hhi(COMMODITY_SHARES[sel])
        k1, k2, k3 = st.columns(3)
        with k1: mcard("HHI (Concentration)", f"{hhi:.4f}")
        with k2: mcard("Export Value", f"${COMMODITY_VALUES_M[sel]:,}M")
        with k3: mcard("Concentration",
                        "High" if hhi > 0.15 else "Moderate" if hhi > 0.10 else "Low")


# ════════════════════════════════════════════════════════════════════
# PAGE: SHOCK SIMULATOR
# ════════════════════════════════════════════════════════════════════

elif page == "⚡ Shock Simulator":
    st.markdown('<div class="main-hdr">Coercion Shock Simulator</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-hdr">Monte Carlo simulation with '
                'uncertainty bands</div>', unsafe_allow_html=True)

    cl, cr = st.columns(2)
    with cl:
        s_com = st.selectbox("Target commodity", COMMODITIES)
        s_pow = st.selectbox("Coercing power", GREAT_POWERS, index=3)
    with cr:
        s_shk = st.selectbox("Shock type", list(COERCION_SHOCKS.keys()))
        s_hor = st.slider("Horizon (months)", 12, 120, 60, 6)

    s_cms = st.multiselect("Active countermeasures", list(COUNTERMEASURES.keys()),
                           default=["AfCFTA Rerouting", "Export Diversification"])

    with st.expander("🧠 Bounded Rationality Parameters", expanded=False):
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            br_att = st.slider("Attention bandwidth", 0.1, 1.0, 0.6, 0.05)
            br_la = st.slider("Loss aversion (λ)", 1.0, 4.0, 2.25, 0.25)
        with bc2:
            br_sq = st.slider("Status-quo bias", 0.0, 0.8, 0.30, 0.05)
            br_sat = st.slider("Satisficing threshold", 0.3, 0.9, 0.60, 0.05)
        with bc3:
            br_dis = st.slider("Discount rate", 0.01, 0.20, 0.08, 0.01)
            br_lr = st.slider("Learning rate", 0.01, 0.15, 0.05, 0.01)

    if st.button("▶ Run Simulation", type="primary", use_container_width=True):
        cfg = SimConfig(commodity=s_com, power=s_pow, shock=s_shk,
                        horizon=s_hor, n_mc=n_mc, cms=s_cms,
                        state_cap=sc_override,
                        br=BRParams(br_att, br_la, br_sq, br_sat, br_dis, br_lr))

        with st.spinner(f"Running {n_mc} Monte Carlo paths..."):
            summary, term = run_mc(cfg, seed=42)

        st.markdown("---")
        k1, k2, k3, k4 = st.columns(4)
        with k1: mcard("Mean Terminal", f"{term['Resilience'].mean():.1%}")
        with k2: mcard("Worst (P5)", f"{term['Resilience'].quantile(0.05):.1%}")
        with k3: mcard("Best (P95)", f"{term['Resilience'].quantile(0.95):.1%}")
        with k4: mcard("Cum. Loss", f"${summary['Mean_Loss'].sum():,.0f}M")

        # Fan chart
        st.markdown("#### Resilience Trajectory with Uncertainty Bands")
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=pd.concat([summary["Month"], summary["Month"][::-1]]),
            y=pd.concat([summary["P95"], summary["P5"][::-1]]),
            fill="toself", fillcolor="rgba(32,128,141,0.12)",
            line=dict(color="rgba(0,0,0,0)"), name="P5-P95"))
        fig.add_trace(go.Scatter(
            x=pd.concat([summary["Month"], summary["Month"][::-1]]),
            y=pd.concat([summary["P75"], summary["P25"][::-1]]),
            fill="toself", fillcolor="rgba(32,128,141,0.25)",
            line=dict(color="rgba(0,0,0,0)"), name="P25-P75"))
        fig.add_trace(go.Scatter(
            x=summary["Month"], y=summary["P50"],
            line=dict(color="#20808D", width=3), name="Median"))
        fig.add_hline(y=1.0, line_dash="dash", line_color="#999",
                      annotation_text="Baseline")
        fig.add_hline(y=0.50, line_dash="dot", line_color="#B91C1C",
                      annotation_text="Critical (50%)")
        fig.update_layout(height=420, margin=dict(l=20,r=20,t=30,b=20),
                          yaxis_title="Resilience Ratio",
                          xaxis_title="Months After Shock",
                          yaxis_tickformat=".0%",
                          font=dict(family="Inter"),
                          legend=dict(orientation="h", yanchor="bottom",
                                      y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        # Decomposition
        st.markdown("#### Loss & Recovery Decomposition")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=summary["Month"], y=summary["Mean_Loss"],
            fill="tozeroy", name="Mean Loss ($M)",
            fillcolor="rgba(185,28,28,0.2)", line=dict(color="#B91C1C")))
        fig2.add_trace(go.Scatter(
            x=summary["Month"], y=summary["Mean_Rec"],
            fill="tozeroy", name="Mean Recovery ($M)",
            fillcolor="rgba(5,150,105,0.2)", line=dict(color="#059669")))
        fig2.add_trace(go.Scatter(
            x=summary["Month"], y=summary["Mean_Gap"],
            mode="lines", name="Residual Gap ($M)",
            line=dict(color="#D97706", width=2, dash="dash")))
        fig2.update_layout(height=350, margin=dict(l=20,r=20,t=20,b=20),
                           yaxis_title="USD Millions",
                           xaxis_title="Months After Shock",
                           font=dict(family="Inter"),
                           legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig2, use_container_width=True)

        # Terminal histogram
        st.markdown("#### Terminal Resilience Distribution")
        fig3 = px.histogram(term, x="Resilience", nbins=40,
                            color_discrete_sequence=["#20808D"], marginal="box")
        fig3.add_vline(x=term["Resilience"].mean(), line_dash="dash",
                       line_color="#A84B2F",
                       annotation_text=f'Mean: {term["Resilience"].mean():.3f}')
        fig3.update_layout(height=300, margin=dict(l=20,r=20,t=30,b=20),
                           xaxis_title="Terminal Resilience",
                           xaxis_tickformat=".0%", font=dict(family="Inter"))
        st.plotly_chart(fig3, use_container_width=True)

        st.download_button("📥 Export Results CSV", summary.to_csv(index=False),
                           f"sim_{s_com}_{s_pow}.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════
# PAGE: COUNTERMEASURE LAB
# ════════════════════════════════════════════════════════════════════

elif page == "🛡️ Countermeasure Lab":
    st.markdown('<div class="main-hdr">Countermeasure Laboratory</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-hdr">A/B portfolio comparison with path-dependence '
                'dynamics</div>', unsafe_allow_html=True)

    # Catalogue
    st.markdown("#### Countermeasure Catalog")
    cat = []
    for n, i in COUNTERMEASURES.items():
        cat.append({"Name": n, "Recovery %": f"{i['trade_recovery_pct']:.0%}",
                     "Time": f"{i['time_to_effect_months']} mo",
                     "Cost (% GDP)": f"{i['cost_gdp_pct']:.1%}",
                     "Cap Threshold": f"{i['state_capacity_threshold']:.0%}",
                     "Path Dep.": f"{i['path_dependence_factor']:.1f}",
                     "Description": i["description"]})
    st.dataframe(pd.DataFrame(cat), use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("#### A/B Portfolio Comparison")

    cc1, cc2 = st.columns(2)
    with cc1:
        cm_com = st.selectbox("Commodity", COMMODITIES, key="cml_c")
        cm_pow = st.selectbox("Coercing power", GREAT_POWERS, key="cml_p", index=3)
    with cc2:
        cm_shk = st.selectbox("Shock type", list(COERCION_SHOCKS.keys()), key="cml_s")

    ab1, ab2 = st.columns(2)
    with ab1:
        st.markdown("**Portfolio A**")
        pa = st.multiselect("Measures (A)", list(COUNTERMEASURES.keys()),
                            default=["AfCFTA Rerouting", "Export Diversification"],
                            key="pa")
    with ab2:
        st.markdown("**Portfolio B**")
        pb = st.multiselect("Measures (B)", list(COUNTERMEASURES.keys()),
                            default=["Value-Chain Upgrading",
                                     "WTO Complaint / Legal Signaling"],
                            key="pb")

    if st.button("▶ Compare Portfolios", type="primary", use_container_width=True):
        with st.spinner("Running A/B comparison..."):
            sa, ta = run_mc(SimConfig(commodity=cm_com, power=cm_pow,
                                     shock=cm_shk, cms=pa, n_mc=n_mc,
                                     state_cap=sc_override), 42)
            sb_, tb = run_mc(SimConfig(commodity=cm_com, power=cm_pow,
                                      shock=cm_shk, cms=pb, n_mc=n_mc,
                                      state_cap=sc_override), 42)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=sa["Month"], y=sa["P50"],
                                 line=dict(color="#20808D", width=3),
                                 name="A (Median)"))
        fig.add_trace(go.Scatter(
            x=pd.concat([sa["Month"], sa["Month"][::-1]]),
            y=pd.concat([sa["P75"], sa["P25"][::-1]]),
            fill="toself", fillcolor="rgba(32,128,141,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="A (IQR)"))
        fig.add_trace(go.Scatter(x=sb_["Month"], y=sb_["P50"],
                                 line=dict(color="#A84B2F", width=3),
                                 name="B (Median)"))
        fig.add_trace(go.Scatter(
            x=pd.concat([sb_["Month"], sb_["Month"][::-1]]),
            y=pd.concat([sb_["P75"], sb_["P25"][::-1]]),
            fill="toself", fillcolor="rgba(168,75,47,0.15)",
            line=dict(color="rgba(0,0,0,0)"), name="B (IQR)"))
        fig.add_hline(y=1.0, line_dash="dash", line_color="#999")
        fig.update_layout(height=420, yaxis_title="Resilience Ratio",
                          xaxis_title="Months After Shock",
                          yaxis_tickformat=".0%", font=dict(family="Inter"),
                          margin=dict(l=20,r=20,t=30,b=20),
                          legend=dict(orientation="h", yanchor="bottom",
                                      y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig, use_container_width=True)

        comp = pd.DataFrame({
            "Metric": ["Mean Terminal", "P5 (Worst)", "P95 (Best)", "Cost (% GDP)"],
            "Portfolio A": [
                f"{ta['Resilience'].mean():.1%}",
                f"{ta['Resilience'].quantile(0.05):.1%}",
                f"{ta['Resilience'].quantile(0.95):.1%}",
                f"{sum(COUNTERMEASURES[c]['cost_gdp_pct'] for c in pa):.2%}"],
            "Portfolio B": [
                f"{tb['Resilience'].mean():.1%}",
                f"{tb['Resilience'].quantile(0.05):.1%}",
                f"{tb['Resilience'].quantile(0.95):.1%}",
                f"{sum(COUNTERMEASURES[c]['cost_gdp_pct'] for c in pb):.2%}"],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: COERCION LADDER
# ════════════════════════════════════════════════════════════════════

elif page == "🪜 Coercion Ladder":
    st.markdown('<div class="main-hdr">Coercion Ladder Heatmap</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-hdr">Escalation thresholds across all great '
                'powers and shock types</div>', unsafe_allow_html=True)

    with st.expander("📋 Shock Catalog", expanded=False):
        st.dataframe(get_coercion_ladder(), use_container_width=True,
                     hide_index=True)

    cl_com = st.selectbox("Commodity", COMMODITIES, key="cl_c")
    cl_cms = st.multiselect("Active countermeasures", list(COUNTERMEASURES.keys()),
                            default=["AfCFTA Rerouting", "Export Diversification"],
                            key="cl_cm")

    if st.button("▶ Generate Coercion Ladder", type="primary",
                 use_container_width=True):
        with st.spinner("Computing escalation thresholds (4 powers x 6 shocks)..."):
            esc = compute_escalation(cl_com, cl_cms, min(n_mc, 150), 42)

        piv = esc.pivot(index="Shock Type", columns="Great Power",
                        values="Mean Terminal Resilience")
        order = piv.mean(axis=1).sort_values().index.tolist()
        piv = piv.loc[order]

        st.markdown(f"#### Escalation Heatmap: {cl_com}")
        fig = px.imshow(
            piv.values, x=piv.columns.tolist(), y=piv.index.tolist(),
            color_continuous_scale=[[0,"#67000D"],[0.3,"#D32F2F"],
                                    [0.5,"#FF8F00"],[0.7,"#FFC553"],
                                    [0.85,"#BCE2E7"],[1,"#E8F5E9"]],
            text_auto=".1%", aspect="auto", zmin=0.3, zmax=1.0)
        fig.update_layout(height=450, margin=dict(l=20,r=20,t=30,b=20),
                          font=dict(family="Inter",size=12),
                          coloraxis_colorbar=dict(title="Resilience",
                                                  tickformat=".0%"),
                          xaxis_title="Coercing Great Power",
                          yaxis_title="Escalation Level (most severe → least)")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('<p class="sn">Cells = mean terminal resilience after 48 months. '
                    'Red = critical; green = robust.</p>', unsafe_allow_html=True)

        st.markdown("#### Detailed Risk Assessment")
        disp = esc.copy()
        disp["Mean Terminal Resilience"] = disp["Mean Terminal Resilience"].map(
            lambda x: f"{x:.1%}")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        st.download_button("📥 Export Ladder CSV", esc.to_csv(index=False),
                           f"ladder_{cl_com}.csv", "text/csv")


# ════════════════════════════════════════════════════════════════════
# PAGE: PORTFOLIO RANKINGS
# ════════════════════════════════════════════════════════════════════

elif page == "📈 Portfolio Rankings":
    st.markdown('<div class="main-hdr">Resilience Portfolio Rankings</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-hdr">Cost-adjusted countermeasure portfolio '
                'rankings</div>', unsafe_allow_html=True)

    pr1, pr2, pr3 = st.columns(3)
    with pr1: pr_com = st.selectbox("Commodity", COMMODITIES, key="pr_c")
    with pr2: pr_pow = st.selectbox("Power", GREAT_POWERS, key="pr_p", index=3)
    with pr3: pr_shk = st.selectbox("Shock", list(COERCION_SHOCKS.keys()), key="pr_s")

    if st.button("▶ Rank Portfolios", type="primary", use_container_width=True):
        with st.spinner("Evaluating all portfolios..."):
            rnk = rank_portfolios(pr_com, pr_pow, pr_shk, n_mc, 60, 42)

        st.markdown("#### Performance Ranking")
        fig = go.Figure()
        fig.add_trace(go.Bar(
            y=rnk["Portfolio"], x=rnk["Mean Terminal"], orientation="h",
            marker_color=[COLORS[i % len(COLORS)] for i in range(len(rnk))],
            text=[f'{v:.1%}' for v in rnk["Mean Terminal"]],
            textposition="outside", name="Mean"))
        fig.add_trace(go.Scatter(
            y=rnk["Portfolio"], x=rnk["P5 (Worst)"],
            mode="markers",
            marker=dict(symbol="line-ew", size=12, color="#B91C1C", line_width=2),
            name="P5"))
        fig.add_trace(go.Scatter(
            y=rnk["Portfolio"], x=rnk["P95 (Best)"],
            mode="markers",
            marker=dict(symbol="line-ew", size=12, color="#059669", line_width=2),
            name="P95"))
        fig.update_layout(height=500, margin=dict(l=20,r=80,t=30,b=20),
                          xaxis_title="Terminal Resilience",
                          xaxis_tickformat=".0%", font=dict(family="Inter"),
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Detailed Rankings")
        disp = rnk.copy()
        for col in ["Mean Terminal", "P5 (Worst)", "P95 (Best)"]:
            disp[col] = disp[col].map(lambda x: f"{x:.1%}")
        disp["Cost (% GDP)"] = disp["Cost (% GDP)"].map(lambda x: f"{x:.2f}%")
        st.dataframe(disp, use_container_width=True, hide_index=True)

        # Efficiency frontier
        st.markdown("#### Efficiency Frontier: Resilience vs Cost")
        fig2 = px.scatter(rnk, x="Cost (% GDP)", y="Mean Terminal",
                          text="Portfolio", color="Mean Terminal",
                          color_continuous_scale=[[0,"#B91C1C"],[0.5,"#FFC553"],
                                                  [1,"#059669"]],
                          size="Cost-Adj Score", size_max=30)
        fig2.update_traces(textposition="top center", textfont_size=10)
        fig2.update_layout(height=400, margin=dict(l=20,r=20,t=30,b=20),
                           yaxis_title="Mean Terminal Resilience",
                           yaxis_tickformat=".0%",
                           xaxis_title="Est. Cost (% GDP)",
                           font=dict(family="Inter"))
        st.plotly_chart(fig2, use_container_width=True)

        st.download_button("📥 Export Rankings Excel", to_excel(rnk),
                           f"rankings_{pr_com}.xlsx",
                           "application/vnd.openxmlformats-officedocument"
                           ".spreadsheetml.sheet")

    st.markdown("---")
    st.markdown("#### Cross-Commodity Comparison")
    if st.button("▶ Run Cross-Commodity Analysis", use_container_width=True):
        with st.spinner("Running cross-commodity analysis..."):
            cross = run_cross_commodity(pr_pow, pr_shk, n_mc=min(n_mc, 200), seed=42)

        fig3 = go.Figure()
        for i, row in cross.iterrows():
            fig3.add_trace(go.Bar(
                x=[row["Commodity"]], y=[row["Mean Terminal"]],
                name=row["Commodity"],
                marker_color=COLORS[i % len(COLORS)],
                text=f'{row["Mean Terminal"]:.1%}', textposition="outside",
                error_y=dict(type="data", symmetric=False,
                             array=[row["P95"] - row["Mean Terminal"]],
                             arrayminus=[row["Mean Terminal"] - row["P5"]])))
        fig3.update_layout(height=400, yaxis_title="Terminal Resilience",
                           yaxis_tickformat=".0%", showlegend=False,
                           font=dict(family="Inter"),
                           margin=dict(l=20,r=20,t=30,b=20))
        st.plotly_chart(fig3, use_container_width=True)
        st.dataframe(cross, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════════════════════════════
# PAGE: METHODOLOGY & SOURCES
# ════════════════════════════════════════════════════════════════════

elif page == "📖 Methodology & Sources":
    st.markdown('<div class="main-hdr">Methodology & Sources</div>',
                unsafe_allow_html=True)

    st.markdown("""
### Theoretical Framework

**1. Commodity Weaponisation Literature**
- Farrell & Newman (2019): "Weaponized Interdependence" — network centrality enables coercion
- Drezner (2021): sanctions efficacy and asymmetric power in commodity markets
- Bown (2022): export controls and supply-chain coercion in the US-China context
- SIPRI (2026): Resource mercantilism and great-power perils for the global rest

**2. Bounded Rationality & Decision Theory**
- Kahneman & Tversky (1979): Prospect theory — loss aversion (lambda ~ 2.25)
- Simon (1956): Satisficing under uncertainty
- March (1991): Exploration vs exploitation in organisational adaptation
- Arthur (1989): Path dependence and lock-in effects

**3. Small-State Strategic Studies**
- Keohane (1969): "Lilliputian's Dilemmas"
- Thorhallsson (2012): Small-state shelter theory
- UNCTAD EDAR 2024: AfCFTA trade resilience potential

---

### Data Sources

| Domain | Source | Year |
|---|---|---|
| Trade flows (cocoa, gold, oil) | UN Comtrade via WITS | 2023 |
| FDI stocks by country | World Bank / UNCTAD | 2023 |
| Governance indicators | World Bank WGI | 2023 |
| Port throughput | Ghana Ports & Harbours Authority | 2023-24 |
| Maritime profiles | UNCTAD Data Hub | 2023 |
| Gold exports | Ghana Gold Board (GoldBod) | 2023 |
| Shock precedents | Brookings, UNCTAD, SIPRI | 2022-26 |

---

### Exposure Index

**E_ij = w1 x TradeShare_ij + w2 x FDIShare_ij + w3 x LogisticsProxy_ij**

Default weights: Trade 45%, FDI 30%, Logistics 25% (adjustable via sidebar).

### Vulnerability Score

**V_i = max(E_ij) x HHI_i x (1 - StateCapacity) x (0.5 + GDPShare_i)**

### Simulation Engine

- N Monte Carlo paths (default 200; adjustable 50-1000)
- Monthly timesteps over shock horizon
- Stochastic noise (sigma = 0.08) per step
- Shock dynamics: 3-month ramp-up, plateau, exponential decay (lambda = 0.06/month)
- Cascading amplifier: 1.5 + 0.5 x trade_disruption
- Countermeasure activation: logistic curve centred on time-to-effect
- Bounded-rationality filters: attention, loss aversion, status-quo bias
- Path-dependence penalties, Bayesian learning (5%/period)
- Diminishing returns on combined countermeasures

### Limitations

- Trade data is 2023-vintage; real-time shifts not captured
- FDI sector breakdown is approximate
- Maritime dependency is a composite proxy
- Bounded-rationality parameters from experimental economics, not country-calibrated
- "Others" category may mask significant bilateral relationships
- Stochastic noise and uncertainty bands mitigate false precision

---

### Citation

> Pivot Coercion Resilience Mapper: Commodity Corridor Exposure Simulator
> under Great-Power Rivalry. Data: UN Comtrade 2023, World Bank WGI,
> UNCTAD, GPHA. Methodology: Farrell & Newman (2019),
> Kahneman & Tversky (1979), UNCTAD EDAR 2024.
    """)

    st.markdown("---")
    st.markdown("### Ghana Governance Profile (WGI 2023)")
    wdf = pd.DataFrame([{"Indicator": k, "Percentile": v}
                         for k, v in WGI_SCORES.items()])
    fig = px.bar(wdf, x="Indicator", y="Percentile", color="Percentile",
                 color_continuous_scale=[[0,"#B91C1C"],[0.5,"#FFC553"],[1,"#059669"]],
                 text_auto=".1f")
    fig.update_layout(height=350, yaxis_range=[0,100],
                      margin=dict(l=20,r=20,t=20,b=20),
                      font=dict(family="Inter"), showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(f"""
**Composite State Capacity Index:** {STATE_CAPACITY_INDEX:.3f}

**Port Infrastructure (2023):**
- Tema Port: {PORT_THROUGHPUT['Tema']:.1f} M tonnes
- Takoradi Port: {PORT_THROUGHPUT['Takoradi']:.1f} M tonnes
- Total: {sum(PORT_THROUGHPUT.values()):.1f} M tonnes

Sources: [Ghana Ports & Harbours Authority](https://ghanaports.gov.gh/),
[World Bank WGI](https://www.worldbank.org/en/publication/worldwide-governance-indicators),
[UN Comtrade](https://comtrade.un.org/)
    """)
