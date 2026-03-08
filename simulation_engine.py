"""
simulation_engine.py – Coercion shock simulation with bounded rationality
and path-dependence dynamics.

Implements:
  1. Stochastic coercion shock injection
  2. Agent-based adaptive response model (with bounded rationality)
  3. System-dynamics feedback loops (path dependence, learning curves)
  4. Monte Carlo resilience portfolio ranking
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from data_engine import (
    COMMODITIES, GREAT_POWERS, COMMODITY_VALUES_M,
    COERCION_SHOCKS, COUNTERMEASURES, STATE_CAPACITY_INDEX,
    compute_exposure_matrix, COMMODITY_SHARES,
)

# ──────────────────────────────────────────────────────────────────────
# 1.  BOUNDED RATIONALITY PARAMETERS
# ──────────────────────────────────────────────────────────────────────

@dataclass
class BoundedRationalityParams:
    """
    Parameters governing the state agent's decision-making under uncertainty.
    Based on Kahneman & Tversky (1979) prospect theory + March (1991) exploration/exploitation.
    """
    attention_bandwidth: float = 0.6     # fraction of information processed (0-1)
    loss_aversion_lambda: float = 2.25   # Kahneman-Tversky loss-aversion coefficient
    status_quo_bias: float = 0.30        # inertia toward existing policy
    satisficing_threshold: float = 0.60  # acceptable-enough outcome (Simon 1956)
    discount_rate: float = 0.08          # annual rate for future payoffs
    learning_rate: float = 0.05          # per-period Bayesian updating speed


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""
    commodity: str = "Cocoa"
    coercing_power: str = "European Union"
    shock_type: str = "Export Ban (full)"
    time_horizon_months: int = 60
    n_monte_carlo: int = 500
    stochastic_noise_std: float = 0.08
    state_capacity_override: Optional[float] = None
    countermeasures_selected: List[str] = field(default_factory=lambda: [
        "AfCFTA Rerouting", "Export Diversification"
    ])
    bounded_rationality: BoundedRationalityParams = field(
        default_factory=BoundedRationalityParams
    )


# ──────────────────────────────────────────────────────────────────────
# 2.  SINGLE-RUN SIMULATION
# ──────────────────────────────────────────────────────────────────────

def _apply_shock(
    baseline_value: float,
    shock_params: dict,
    exposure_to_coercer: float,
    month: int,
    noise: float,
) -> float:
    """
    Compute the trade-value impact of a coercion shock at a given month.
    Shock decays exponentially after its peak duration.

    The disruption is amplified beyond raw exposure share to reflect
    cascading effects: supply-chain knock-ons, investor confidence loss,
    insurance and shipping premium spikes, and reputational contagion.
    Amplifier = 1.5 + 0.5 * trade_disruption (severe shocks cascade more).
    """
    duration = shock_params["duration_months"]
    # Amplifier captures cascading / knock-on effects beyond direct trade share
    amplifier = 1.5 + 0.5 * shock_params["trade_disruption"]
    peak_disruption = min(0.95, shock_params["trade_disruption"] * exposure_to_coercer * amplifier)
    price_effect = shock_params["price_impact_pct"]

    # Ramp-up over first 3 months, plateau, then slow decay
    if month <= 3:
        ramp = month / 3.0
    elif month <= duration:
        ramp = 1.0
    else:
        decay_rate = 0.06  # slower decay — coercion effects linger
        ramp = np.exp(-decay_rate * (month - duration))

    disruption = peak_disruption * ramp
    price_adj = 1.0 + price_effect * ramp
    # Also add FDI-flight drag on output
    fdi_drag = shock_params["fdi_flight_pct"] * exposure_to_coercer * 0.15 * ramp
    noisy_disruption = max(0, min(0.95, disruption + fdi_drag + noise))

    shocked_value = baseline_value * (1.0 - noisy_disruption) * price_adj
    return max(0, shocked_value)


def _apply_countermeasures(
    current_value: float,
    baseline_value: float,
    countermeasures: List[str],
    month: int,
    state_capacity: float,
    br_params: BoundedRationalityParams,
    rng: np.random.Generator,
) -> Tuple[float, Dict[str, float]]:
    """
    Apply selected countermeasures with bounded-rationality filtering.
    Returns (recovered_value, {countermeasure: marginal_recovery}).
    """
    gap = baseline_value - current_value
    if gap <= 0:
        return current_value, {}

    total_recovery = 0.0
    cm_contributions = {}

    for cm_name in countermeasures:
        cm = COUNTERMEASURES[cm_name]

        # State-capacity gate
        if state_capacity < cm["state_capacity_threshold"]:
            cm_contributions[cm_name] = 0.0
            continue

        # Time-to-effect logistic curve
        t_half = cm["time_to_effect_months"]
        activation = 1.0 / (1.0 + np.exp(-0.3 * (month - t_half)))

        # Bounded rationality filters
        # 1. Attention: may not fully consider this option
        attention_gate = 1.0 if rng.random() < br_params.attention_bandwidth else 0.4
        # 2. Status-quo bias reduces effectiveness of novel measures
        sq_penalty = 1.0 - br_params.status_quo_bias * cm["path_dependence_factor"]
        # 3. Loss-aversion: overweight downside of costly measures
        cost_perceived = cm["cost_gdp_pct"] * br_params.loss_aversion_lambda
        cost_penalty = max(0.3, 1.0 - cost_perceived * 5)

        effective_recovery = (
            cm["trade_recovery_pct"]
            * activation
            * attention_gate
            * sq_penalty
            * cost_penalty
            * (0.5 + 0.5 * state_capacity)  # capacity multiplier
        )

        # Learning: slight improvement each period
        learning_bonus = 1.0 + br_params.learning_rate * min(month, 36)
        effective_recovery *= learning_bonus

        # Stochastic noise
        effective_recovery *= (1.0 + rng.normal(0, 0.05))
        effective_recovery = max(0, min(0.95, effective_recovery))

        marginal = gap * effective_recovery
        # Diminishing returns on combined countermeasures
        marginal *= max(0.3, 1.0 - total_recovery / gap) if gap > 0 else 0
        total_recovery += marginal
        cm_contributions[cm_name] = marginal

    recovered_value = current_value + min(total_recovery, gap)
    return recovered_value, cm_contributions


def run_single_simulation(
    config: SimulationConfig,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Run one Monte Carlo path.
    Returns monthly time-series: [month, baseline, shocked, recovered, gap, ...].
    """
    baseline = COMMODITY_VALUES_M[config.commodity]
    shock = COERCION_SHOCKS[config.shock_type]
    state_cap = config.state_capacity_override or STATE_CAPACITY_INDEX

    # Compute exposure of this commodity to coercing power
    exposure_matrix = compute_exposure_matrix()
    exposure = exposure_matrix.loc[config.commodity, config.coercing_power]

    records = []
    for m in range(1, config.time_horizon_months + 1):
        noise = rng.normal(0, config.stochastic_noise_std)

        # Apply shock
        shocked_val = _apply_shock(baseline, shock, exposure, m, noise)

        # Apply countermeasures
        recovered_val, cm_contribs = _apply_countermeasures(
            shocked_val, baseline, config.countermeasures_selected,
            m, state_cap, config.bounded_rationality, rng,
        )

        record = {
            "Month": m,
            "Baseline ($M)": baseline,
            "Shocked ($M)": round(shocked_val, 2),
            "Recovered ($M)": round(recovered_val, 2),
            "Loss ($M)": round(baseline - shocked_val, 2),
            "Recovery ($M)": round(recovered_val - shocked_val, 2),
            "Residual Gap ($M)": round(baseline - recovered_val, 2),
            "Resilience Ratio": round(recovered_val / baseline, 4) if baseline > 0 else 0,
        }
        for cm_name, val in cm_contribs.items():
            record[f"CM: {cm_name} ($M)"] = round(val, 2)

        records.append(record)

    return pd.DataFrame(records)


# ──────────────────────────────────────────────────────────────────────
# 3.  MONTE CARLO ENGINE
# ──────────────────────────────────────────────────────────────────────

def run_monte_carlo(
    config: SimulationConfig,
    seed: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run N Monte Carlo simulations.
    Returns:
      - summary_stats: mean/p5/p25/p50/p75/p95 time series
      - all_paths: raw monthly data (long format with run_id)
      - terminal_distribution: distribution of final-period resilience ratios
    """
    rng = np.random.default_rng(seed)
    all_runs = []

    for i in range(config.n_monte_carlo):
        run_df = run_single_simulation(config, rng)
        run_df["Run"] = i
        all_runs.append(run_df)

    all_paths = pd.concat(all_runs, ignore_index=True)

    # Summary statistics per month
    summary = all_paths.groupby("Month").agg(
        Mean_Resilience=("Resilience Ratio", "mean"),
        P5_Resilience=("Resilience Ratio", lambda x: np.percentile(x, 5)),
        P25_Resilience=("Resilience Ratio", lambda x: np.percentile(x, 25)),
        P50_Resilience=("Resilience Ratio", "median"),
        P75_Resilience=("Resilience Ratio", lambda x: np.percentile(x, 75)),
        P95_Resilience=("Resilience Ratio", lambda x: np.percentile(x, 95)),
        Mean_Loss=("Loss ($M)", "mean"),
        Mean_Recovery=("Recovery ($M)", "mean"),
        Mean_Residual_Gap=("Residual Gap ($M)", "mean"),
        Mean_Shocked=("Shocked ($M)", "mean"),
        Mean_Recovered=("Recovered ($M)", "mean"),
    ).reset_index()

    # Terminal distribution
    terminal = all_paths[all_paths["Month"] == config.time_horizon_months].copy()
    terminal_dist = terminal[["Run", "Resilience Ratio", "Residual Gap ($M)", "Recovered ($M)"]].copy()

    return summary, all_paths, terminal_dist


# ──────────────────────────────────────────────────────────────────────
# 4.  MULTI-SECTOR SCENARIO COMPARISON
# ──────────────────────────────────────────────────────────────────────

def run_multi_commodity_comparison(
    coercing_power: str = "European Union",
    shock_type: str = "Export Ban (full)",
    countermeasures: List[str] = None,
    n_mc: int = 200,
    time_horizon: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Run the same shock scenario across all commodities.
    Returns a comparison DataFrame.
    """
    if countermeasures is None:
        countermeasures = ["AfCFTA Rerouting", "Export Diversification"]

    results = []
    for commodity in COMMODITIES:
        cfg = SimulationConfig(
            commodity=commodity,
            coercing_power=coercing_power,
            shock_type=shock_type,
            countermeasures_selected=countermeasures,
            n_monte_carlo=n_mc,
            time_horizon_months=time_horizon,
        )
        summary, _, terminal = run_monte_carlo(cfg, seed=seed)

        results.append({
            "Commodity": commodity,
            "Baseline ($M)": COMMODITY_VALUES_M[commodity],
            "Mean Terminal Resilience": round(terminal["Resilience Ratio"].mean(), 4),
            "P5 Terminal Resilience": round(terminal["Resilience Ratio"].quantile(0.05), 4),
            "P95 Terminal Resilience": round(terminal["Resilience Ratio"].quantile(0.95), 4),
            "Mean Cumulative Loss ($M)": round(
                summary["Mean_Loss"].sum(), 1
            ),
            "Mean Cumulative Recovery ($M)": round(
                summary["Mean_Recovery"].sum(), 1
            ),
        })

    return pd.DataFrame(results)


# ──────────────────────────────────────────────────────────────────────
# 5.  PORTFOLIO RANKING ENGINE
# ──────────────────────────────────────────────────────────────────────

def rank_countermeasure_portfolios(
    commodity: str = "Cocoa",
    coercing_power: str = "European Union",
    shock_type: str = "Export Ban (full)",
    n_mc: int = 200,
    time_horizon: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Test each countermeasure individually + key combinations,
    then rank by terminal resilience.
    """
    portfolios = {name: [name] for name in COUNTERMEASURES.keys()}
    # Add some combined portfolios
    portfolios["AfCFTA + Diversification"] = ["AfCFTA Rerouting", "Export Diversification"]
    portfolios["Full Defensive"] = [
        "AfCFTA Rerouting", "Export Diversification",
        "Strategic Stockpiling", "Hedging via Commodity Futures"
    ]
    portfolios["Legal + Diplomatic"] = [
        "WTO Complaint / Legal Signaling", "Counter-Alignment (Pivot)"
    ]
    portfolios["Industrial Policy"] = [
        "Value-Chain Upgrading", "AfCFTA Rerouting"
    ]
    portfolios["No Response (Baseline)"] = []

    results = []
    for port_name, cms in portfolios.items():
        cfg = SimulationConfig(
            commodity=commodity,
            coercing_power=coercing_power,
            shock_type=shock_type,
            countermeasures_selected=cms,
            n_monte_carlo=n_mc,
            time_horizon_months=time_horizon,
        )
        summary, _, terminal = run_monte_carlo(cfg, seed=seed)

        total_cost = sum(COUNTERMEASURES[c]["cost_gdp_pct"] for c in cms)
        results.append({
            "Portfolio": port_name,
            "Countermeasures": ", ".join(cms) if cms else "(none)",
            "Mean Terminal Resilience": round(terminal["Resilience Ratio"].mean(), 4),
            "P5 (Worst Case)": round(terminal["Resilience Ratio"].quantile(0.05), 4),
            "P95 (Best Case)": round(terminal["Resilience Ratio"].quantile(0.95), 4),
            "Estimated Cost (% GDP)": round(total_cost * 100, 2),
            "Cost-Adjusted Score": round(
                terminal["Resilience Ratio"].mean() / max(total_cost * 100 + 0.1, 0.1), 4
            ),
        })

    df = pd.DataFrame(results).sort_values("Mean Terminal Resilience", ascending=False)
    return df.reset_index(drop=True)


# ──────────────────────────────────────────────────────────────────────
# 6.  ESCALATION THRESHOLD ANALYSIS
# ──────────────────────────────────────────────────────────────────────

def compute_escalation_thresholds(
    commodity: str = "Cocoa",
    countermeasures: List[str] = None,
    n_mc: int = 150,
    seed: int = 42,
) -> pd.DataFrame:
    """
    For each great power × shock type combination, compute the
    terminal resilience. This populates the 'Coercion Ladder' heatmap.
    """
    if countermeasures is None:
        countermeasures = ["AfCFTA Rerouting", "Export Diversification"]

    records = []
    for gp in GREAT_POWERS:
        for shock_name in COERCION_SHOCKS.keys():
            cfg = SimulationConfig(
                commodity=commodity,
                coercing_power=gp,
                shock_type=shock_name,
                countermeasures_selected=countermeasures,
                n_monte_carlo=n_mc,
                time_horizon_months=48,
            )
            _, _, terminal = run_monte_carlo(cfg, seed=seed)
            mean_res = terminal["Resilience Ratio"].mean()
            records.append({
                "Great Power": gp,
                "Shock Type": shock_name,
                "Mean Terminal Resilience": round(mean_res, 4),
                "Risk Level": (
                    "Critical" if mean_res < 0.50
                    else "High" if mean_res < 0.70
                    else "Moderate" if mean_res < 0.85
                    else "Low"
                ),
            })

    return pd.DataFrame(records)
