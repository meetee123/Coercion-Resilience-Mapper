# Pivot Coercion Resilience Mapper

**Commodity Corridor Exposure Simulator under Great-Power Rivalry**

A multi-scenario resilience engine that maps small and middle powers' strategic
exposures in cocoa, gold, oil, and maritime corridors to coercion by
US/China/Russia/EU, then simulates state-level countermeasures accounting for
bounded rationality and path dependence.

## Features

- **Exposure Analysis**: Sector-specific exposure indices from trade, FDI, and logistics data
- **Shock Simulator**: Monte Carlo simulation of coercion shocks with uncertainty bands
- **Countermeasure Lab**: A/B comparison of adaptive response portfolios
- **Coercion Ladder**: Escalation threshold heatmaps across all great powers and shock types
- **Portfolio Rankings**: Cost-adjusted resilience portfolio rankings with efficiency frontiers
- **Full Methodology**: Transparent documentation with academic citations and data provenance

## Data Sources

| Domain | Source | Year |
|--------|--------|------|
| Trade flows | UN Comtrade / WITS | 2023 |
| FDI stocks | World Bank / UNCTAD | 2023 |
| Governance | World Bank WGI | 2023 |
| Port throughput | Ghana Ports & Harbours Authority | 2023–24 |
| Maritime profiles | UNCTAD Data Hub | 2023 |

## Quick Start (Local)

```bash
# Clone or copy the project directory
cd coercion-mapper

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

## Deploy to Streamlit Community Cloud

1. Push the `coercion-mapper/` directory to a public GitHub repository
2. Go to [share.streamlit.io](https://share.streamlit.io/)
3. Click "New app" → point to your repo
4. Set:
   - **Repository**: `your-username/your-repo`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click "Deploy"

No secrets are required — all data is embedded and derived from public sources.

## Project Structure

```
coercion-mapper/
├── app.py                  # Main Streamlit application (UI + pages)
├── data_engine.py          # Baseline data, exposure indices, shock definitions
├── simulation_engine.py    # Monte Carlo engine, bounded rationality, portfolio ranking
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── .streamlit/
    └── config.toml         # Streamlit theme configuration
```

## Theoretical Foundations

- **Farrell & Newman (2019)**: Weaponized Interdependence
- **Kahneman & Tversky (1979)**: Prospect Theory (loss aversion λ ≈ 2.25)
- **Simon (1956)**: Satisficing under uncertainty
- **UNCTAD EDAR 2024**: AfCFTA trade resilience
- **SIPRI (2026)**: Resource mercantilism and great-power perils

## License

Open source — intended for research, policy analysis, and educational purposes.
