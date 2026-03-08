"""
data_engine.py – Baseline data construction and exposure-index computation
for the Pivot Coercion Resilience Mapper.

Uses calibrated values derived from:
  • UN Comtrade / WITS 2023 (cocoa, gold, oil trade flows)
  • World Bank WGI 2023 (governance proxies)
  • UNCTAD maritime profiles / GPHA annual reports
  • IMF/World Bank FDI statistics
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple

# ──────────────────────────────────────────────────────────────────────
# 1. BASELINE TRADE-FLOW MATRICES  (calibrated to 2023 UN Comtrade)
# ──────────────────────────────────────────────────────────────────────

COMMODITIES = ["Cocoa", "Gold", "Oil", "Maritime Corridor"]
GREAT_POWERS = ["United States", "China", "Russia", "European Union"]

# Ghana cocoa exports 2023: ~$1.1 B total
# Top partners: Netherlands $264 M (EU), US $145 M, Malaysia $140 M,
#   Belgium $131 M (EU), Japan $86 M, France $42 M (EU), Germany $42 M (EU)
COCOA_EXPORT_SHARES = {
    "European Union": 0.52,    # NL+BE+FR+DE+IT+ES+UK≈
    "United States": 0.13,
    "China": 0.01,             # ~$6.7 M
    "Russia": 0.00,
    "Others": 0.34,            # MY, JP, TR, CA, etc.
}

# Ghana gold exports 2023: ~$7.6 B (semi-manufactured) + $15.6 B total
# Switzerland $3.1 B, UAE $1.7 B, South Africa $1.7 B, India $950 M
GOLD_EXPORT_SHARES = {
    "European Union": 0.41,    # Switzerland (not EU but treated as European corridor)
    "United States": 0.02,
    "China": 0.01,
    "Russia": 0.00,
    "Others": 0.56,            # UAE, SA, India, Turkey
}

# Ghana oil: primarily Jubilee & TEN fields; exports ~crude
# Key partners: China ~25%, India ~15%, EU ~20%, US ~5%
OIL_EXPORT_SHARES = {
    "European Union": 0.20,
    "United States": 0.05,
    "China": 0.25,
    "Russia": 0.00,
    "Others": 0.50,
}

# Maritime corridor dependency (proxy: port investment, shipping lines, naval presence)
# Tema/Takoradi throughput 26 M tonnes (2023): Maersk (EU), COSCO (CN), etc.
MARITIME_SHARES = {
    "European Union": 0.40,    # Maersk, MSC, CMA-CGM
    "United States": 0.10,     # US naval presence / Gulf of Guinea patrols
    "China": 0.30,             # COSCO, port investment bids
    "Russia": 0.05,            # Limited Black Sea–Mediterranean rerouting
    "Others": 0.15,
}

COMMODITY_SHARES = {
    "Cocoa": COCOA_EXPORT_SHARES,
    "Gold": GOLD_EXPORT_SHARES,
    "Oil": OIL_EXPORT_SHARES,
    "Maritime Corridor": MARITIME_SHARES,
}

# Absolute export values (USD millions, 2023 baseline)
COMMODITY_VALUES_M = {
    "Cocoa": 1_107,
    "Gold": 7_632,
    "Oil": 4_200,
    "Maritime Corridor": 2_600,  # imputed logistics-services GDP proxy
}

# ──────────────────────────────────────────────────────────────────────
# 2. FDI STOCK SHARES  (calibrated to World Bank / UNCTAD)
# ──────────────────────────────────────────────────────────────────────
FDI_SHARES = {
    "Cocoa": {"European Union": 0.35, "United States": 0.15, "China": 0.05, "Russia": 0.00, "Others": 0.45},
    "Gold": {"European Union": 0.25, "United States": 0.10, "China": 0.20, "Russia": 0.02, "Others": 0.43},
    "Oil": {"European Union": 0.30, "United States": 0.20, "China": 0.25, "Russia": 0.00, "Others": 0.25},
    "Maritime Corridor": {"European Union": 0.35, "United States": 0.10, "China": 0.35, "Russia": 0.05, "Others": 0.15},
}

# ──────────────────────────────────────────────────────────────────────
# 3. STATE-CAPACITY / GOVERNANCE PROXIES  (WGI 2023 + Legatum)
# ──────────────────────────────────────────────────────────────────────
# Ghana WGI percentile ranks (0-100):
WGI_SCORES = {
    "Voice & Accountability": 62.6,
    "Political Stability": 46.2,
    "Government Effectiveness": 43.8,
    "Regulatory Quality": 50.5,
    "Rule of Law": 53.3,
    "Control of Corruption": 41.0,
}

STATE_CAPACITY_INDEX = np.mean(list(WGI_SCORES.values())) / 100.0  # ~0.496

# Port throughput (million tonnes, 2023)
PORT_THROUGHPUT = {"Tema": 18.0, "Takoradi": 8.0}

# ──────────────────────────────────────────────────────────────────────
# 4. EXPOSURE INDEX COMPUTATION
# ──────────────────────────────────────────────────────────────────────

def compute_exposure_matrix(
    trade_weight: float = 0.45,
    fdi_weight: float = 0.30,
    logistics_weight: float = 0.25,
) -> pd.DataFrame:
    """
    Construct the Sector × Great-Power Exposure Matrix.

    Exposure_ij = w_trade * TradeShare_ij + w_fdi * FDI_ij + w_logistics * LogisticsProxy_ij

    Returns a DataFrame with commodities as rows and great powers as columns.
    """
    rows = []
    for commodity in COMMODITIES:
        row = {}
        for gp in GREAT_POWERS:
            trade_s = COMMODITY_SHARES[commodity].get(gp, 0.0)
            fdi_s = FDI_SHARES[commodity].get(gp, 0.0)
            logistic_s = MARITIME_SHARES.get(gp, 0.0) if commodity == "Maritime Corridor" else trade_s * 0.8
            exposure = (trade_weight * trade_s
                        + fdi_weight * fdi_s
                        + logistics_weight * logistic_s)
            row[gp] = round(exposure, 4)
        rows.append(row)

    df = pd.DataFrame(rows, index=COMMODITIES)
    return df


def compute_concentration_hhi(shares: Dict[str, float]) -> float:
    """Herfindahl-Hirschman Index for partner concentration."""
    vals = [v for k, v in shares.items() if k != "Others"]
    return sum(v**2 for v in vals)


def compute_sector_vulnerability(exposure_df: pd.DataFrame) -> pd.DataFrame:
    """
    Augment exposure matrix with:
      - HHI concentration score per commodity
      - GDP-share weight
      - State-capacity adjustment
    """
    total_exports = sum(COMMODITY_VALUES_M.values())
    records = []
    for commodity in COMMODITIES:
        hhi = compute_concentration_hhi(COMMODITY_SHARES[commodity])
        gdp_share = COMMODITY_VALUES_M[commodity] / total_exports
        max_exposure = exposure_df.loc[commodity].max()
        # Vulnerability = max_exposure * HHI * (1 - state_capacity) * gdp_share_weight
        vulnerability = max_exposure * hhi * (1.0 - STATE_CAPACITY_INDEX) * (0.5 + gdp_share)
        records.append({
            "Commodity": commodity,
            "HHI": round(hhi, 4),
            "GDP Share": round(gdp_share, 4),
            "Max Single-Power Exposure": round(max_exposure, 4),
            "Most Exposed To": exposure_df.loc[commodity].idxmax(),
            "Vulnerability Score": round(vulnerability, 4),
        })
    return pd.DataFrame(records).set_index("Commodity")


# ──────────────────────────────────────────────────────────────────────
# 5. COERCION SHOCK DEFINITIONS
# ──────────────────────────────────────────────────────────────────────

COERCION_SHOCKS = {
    "Export Ban (full)": {
        "description": "Great power imposes a complete ban on imports of the commodity from Ghana.",
        "trade_disruption": 1.0,
        "fdi_flight_pct": 0.60,
        "price_impact_pct": -0.30,
        "duration_months": 12,
        "historical_precedent": "Russian grain export restrictions 2022; US sanctions on Iranian oil",
    },
    "Price Cap / Buyer Cartel": {
        "description": "Coordinated price ceiling imposed by major buyers.",
        "trade_disruption": 0.40,
        "fdi_flight_pct": 0.20,
        "price_impact_pct": -0.25,
        "duration_months": 18,
        "historical_precedent": "G7 Russian oil price cap ($60/bbl) Dec 2022",
    },
    "Port Pressure / Logistics Denial": {
        "description": "Denial of port services, insurance, or shipping access.",
        "trade_disruption": 0.70,
        "fdi_flight_pct": 0.30,
        "price_impact_pct": -0.15,
        "duration_months": 6,
        "historical_precedent": "Houthi Red Sea shipping disruption 2023–2024",
    },
    "Sanctions on Key Entities": {
        "description": "Targeted sanctions on state trading companies or mining firms.",
        "trade_disruption": 0.50,
        "fdi_flight_pct": 0.40,
        "price_impact_pct": -0.10,
        "duration_months": 24,
        "historical_precedent": "US OFAC sanctions on Venezuelan PDVSA; EU sanctions on Russian gold",
    },
    "FDI Weaponisation": {
        "description": "Withdrawal or freezing of FDI as political leverage.",
        "trade_disruption": 0.20,
        "fdi_flight_pct": 0.80,
        "price_impact_pct": -0.05,
        "duration_months": 36,
        "historical_precedent": "China's investment freezes in Australia 2020–2021",
    },
    "Diplomatic Pressure (Soft Coercion)": {
        "description": "Implicit threats tied to aid, debt restructuring, or votes in international fora.",
        "trade_disruption": 0.10,
        "fdi_flight_pct": 0.10,
        "price_impact_pct": -0.05,
        "duration_months": 6,
        "historical_precedent": "Chinese 'debt-trap diplomacy' debates; US AGOA conditionality",
    },
}


# ──────────────────────────────────────────────────────────────────────
# 6. COUNTERMEASURE DEFINITIONS
# ──────────────────────────────────────────────────────────────────────

COUNTERMEASURES = {
    "AfCFTA Rerouting": {
        "description": "Redirect exports through AfCFTA partner corridors to reduce great-power dependency.",
        "trade_recovery_pct": 0.40,
        "time_to_effect_months": 12,
        "cost_gdp_pct": 0.02,
        "state_capacity_threshold": 0.40,
        "path_dependence_factor": 0.3,  # moderate lock-in
    },
    "Export Diversification": {
        "description": "Cultivate alternative buyer markets (e.g., ASEAN, Gulf, India).",
        "trade_recovery_pct": 0.55,
        "time_to_effect_months": 18,
        "cost_gdp_pct": 0.03,
        "state_capacity_threshold": 0.45,
        "path_dependence_factor": 0.5,
    },
    "Value-Chain Upgrading": {
        "description": "Process raw commodities domestically (e.g., cocoa → chocolate, gold refining).",
        "trade_recovery_pct": 0.30,
        "time_to_effect_months": 36,
        "cost_gdp_pct": 0.05,
        "state_capacity_threshold": 0.50,
        "path_dependence_factor": 0.7,
    },
    "WTO Complaint / Legal Signaling": {
        "description": "File WTO dispute or invoke bilateral investment treaty arbitration.",
        "trade_recovery_pct": 0.15,
        "time_to_effect_months": 24,
        "cost_gdp_pct": 0.005,
        "state_capacity_threshold": 0.35,
        "path_dependence_factor": 0.1,
    },
    "Strategic Stockpiling": {
        "description": "Build buffer stocks to ride out temporary supply disruptions.",
        "trade_recovery_pct": 0.20,
        "time_to_effect_months": 6,
        "cost_gdp_pct": 0.04,
        "state_capacity_threshold": 0.30,
        "path_dependence_factor": 0.2,
    },
    "Counter-Alignment (Pivot)": {
        "description": "Shift strategic alignment toward a rival great power for protection.",
        "trade_recovery_pct": 0.60,
        "time_to_effect_months": 6,
        "cost_gdp_pct": 0.01,
        "state_capacity_threshold": 0.25,
        "path_dependence_factor": 0.8,  # highest lock-in
    },
    "Hedging via Commodity Futures": {
        "description": "Use financial instruments to lock in prices and reduce volatility exposure.",
        "trade_recovery_pct": 0.10,
        "time_to_effect_months": 3,
        "cost_gdp_pct": 0.01,
        "state_capacity_threshold": 0.45,
        "path_dependence_factor": 0.1,
    },
}


# ──────────────────────────────────────────────────────────────────────
# 7. HELPER: build full baseline snapshot
# ──────────────────────────────────────────────────────────────────────

def build_baseline_snapshot(
    trade_w: float = 0.45,
    fdi_w: float = 0.30,
    logistics_w: float = 0.25,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (exposure_matrix, vulnerability_table)."""
    exposure = compute_exposure_matrix(trade_w, fdi_w, logistics_w)
    vulnerability = compute_sector_vulnerability(exposure)
    return exposure, vulnerability


def get_coercion_ladder_data() -> pd.DataFrame:
    """Return coercion shocks as a DataFrame sorted by escalation severity."""
    records = []
    for name, info in COERCION_SHOCKS.items():
        records.append({
            "Shock Type": name,
            "Trade Disruption": info["trade_disruption"],
            "FDI Flight %": info["fdi_flight_pct"],
            "Price Impact %": info["price_impact_pct"],
            "Duration (months)": info["duration_months"],
            "Description": info["description"],
            "Historical Precedent": info["historical_precedent"],
            "Escalation Score": round(
                0.4 * info["trade_disruption"]
                + 0.3 * info["fdi_flight_pct"]
                + 0.2 * abs(info["price_impact_pct"])
                + 0.1 * min(info["duration_months"] / 36, 1.0),
                3
            ),
        })
    df = pd.DataFrame(records).sort_values("Escalation Score", ascending=True)
    return df.reset_index(drop=True)
