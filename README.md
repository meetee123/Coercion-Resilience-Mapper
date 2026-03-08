# Pivot Coercion Resilience Mapper

**Commodity Corridor Exposure Simulator under Great-Power Rivalry**

A multi-scenario resilience engine that maps small and middle powers' strategic exposures in cocoa, gold, oil, and maritime corridors to coercion by US/China/Russia/EU, then simulates state-level countermeasures accounting for bounded rationality and path dependence.

---

## What It Does

This tool constructs sector-specific exposure indices from trade, FDI, and logistics data; injects coercion shocks (export bans, price caps, port pressure); and runs Monte Carlo simulations of adaptive responses (diversification timing, hedging via AfCFTA rerouting, signaling via WTO complaints). It outputs ranked resilience portfolios and "coercion ladder" heatmaps showing escalation thresholds.

### Pages

| Page | Purpose |
|---|---|
| **Executive Dashboard** | KPI cards, exposure heatmap, vulnerability ranking, trade sunburst |
| **Exposure Analysis** | Full exposure matrix, sectoral deep dives (trade partners, FDI composition, HHI) |
| **Shock Simulator** | Monte Carlo fan charts with P5/P25/P50/P75/P95 uncertainty bands, loss-recovery decomposition, terminal distribution |
| **Countermeasure Lab** | A/B portfolio comparison with side-by-side IQR trajectories |
| **Coercion Ladder** | Escalation heatmap — shock severity × great power, color-coded from critical to low |
| **Portfolio Rankings** | Cost-adjusted scoring, efficiency frontier scatter, cross-commodity comparison |
| **Methodology & Sources** | Academic citations, data provenance, model specification, limitations |

---

## Data Sources

| Domain | Source | Year |
|---|---|---|
| Trade flows (cocoa, gold, oil) | UN Comtrade via WITS | 2023 |
| FDI stocks by country | World Bank / UNCTAD | 2023 |
| Governance indicators | World Bank WGI | 2023 |
| Port throughput | Ghana Ports & Harbours Authority | 2023–24 |
| Maritime profiles | UNCTAD Data Hub | 2023 |
| Gold exports | Ghana Gold Board (GoldBod) | 2023 |
| Shock precedents | Brookings, UNCTAD, SIPRI | 2022–26 |

---

## Theoretical Foundations

- **Farrell & Newman (2019)**: Weaponized Interdependence — network centrality enables coercion
- **Kahneman & Tversky (1979)**: Prospect Theory — loss aversion (λ ≈ 2.25) in policy decisions
- **Simon (1956)**: Satisficing under uncertainty — states accept "good enough" outcomes
- **March (1991)**: Exploration vs. exploitation in organisational adaptation
- **Arthur (1989)**: Path dependence and increasing returns
- **UNCTAD EDAR 2024**: AfCFTA trade resilience and the $3.4 trillion opportunity
- **SIPRI (2026)**: Resource mercantilism and great-power perils for the global rest

---

## Model Specification

### Exposure Index

```
E_ij = w_trade × TradeShare_ij + w_fdi × FDIShare_ij + w_logistics × LogisticsProxy_ij
```

Default weights: Trade 45%, FDI 30%, Logistics 25% (adjustable via sidebar).

### Vulnerability Score

```
V_i = max(E_ij) × HHI_i × (1 − StateCapacity) × (0.5 + GDPShare_i)
```

### Simulation Engine

- **N** Monte Carlo paths (default 200; adjustable 50–1,000)
- Monthly timesteps with stochastic noise (σ = 0.08)
- Shock dynamics: 3-month ramp-up → plateau → exponential decay (λ = 0.06/month)
- Cascading amplifier: 1.5 + 0.5 × trade_disruption (supply-chain knock-ons)
- FDI-flight drag: fdi_flight_pct × exposure × 0.15
- Countermeasure activation: logistic curve centred on time-to-effect
- Bounded-rationality filters: attention bandwidth, loss aversion, status-quo bias
- Path-dependence penalties and Bayesian learning (5%/period)
- Diminishing returns on combined countermeasures

### Coercion Shocks (6 types)

| Shock | Trade Disruption | FDI Flight | Price Impact | Duration |
|---|---|---|---|---|
| Export Ban (full) | 100% | 60% | −30% | 12 mo |
| Price Cap / Buyer Cartel | 40% | 20% | −25% | 18 mo |
| Port Pressure / Logistics Denial | 70% | 30% | −15% | 6 mo |
| Sanctions on Key Entities | 50% | 40% | −10% | 24 mo |
| FDI Weaponisation | 20% | 80% | −5% | 36 mo |
| Diplomatic Pressure (Soft) | 10% | 10% | −5% | 6 mo |

### Countermeasures (7 types)

AfCFTA Rerouting, Export Diversification, Value-Chain Upgrading, WTO Complaint / Legal Signaling, Strategic Stockpiling, Counter-Alignment (Pivot), Hedging via Commodity Futures.

---

## Quick Start (Local)

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Click **New app** → select your repo → set main file to `app.py`
4. **Important**: Click **Advanced settings** and set **Python version to 3.12**
   (Streamlit Cloud defaults to 3.13, which lacks stable wheels for some packages)
5. Click **Deploy**

No secrets required — all data is embedded from public sources.

---

## Project Structure

```
├── app.py                  # Single-file application (data + simulation + UI)
├── requirements.txt        # 5 dependencies, exact versions pinned
├── README.md               # This file
└── .streamlit/
    └── config.toml         # Theme configuration
```

## Dependencies

```
streamlit==1.41.1
pandas==2.2.3
numpy==1.26.4
plotly==5.24.1
scipy==1.14.1
```

---

## Known Limitations

- Trade-share data is 2023-vintage; real-time shifts not captured
- FDI sector breakdown is approximate (country-level disaggregated by proxy)
- Maritime corridor dependency is a composite proxy, not direct measurement
- Bounded-rationality parameters from experimental economics, not country-calibrated
- "Others" trade category may mask significant bilateral relationships
- All results include stochastic uncertainty bands to mitigate false precision

---

## Primary Audiences

- Ministries of Foreign Affairs & Trade
- African Union geoeconomics units
- EU External Action Service (Africa desk)
- Geopolitical risk and compliance teams at commodity traders
- Think tanks (Chatham House, IISS, Brookings, SIPRI)

---

## License

Open source — intended for research, policy analysis, and educational purposes.

## Citation

> Pivot Coercion Resilience Mapper: Commodity Corridor Exposure Simulator
> under Great-Power Rivalry. Data: UN Comtrade 2023, World Bank WGI,
> UNCTAD, GPHA. Methodology: Farrell & Newman (2019),
> Kahneman & Tversky (1979), UNCTAD EDAR 2024.
