# SHIFT – Steel & Hydrogen Integrated Freight Trade

This repository contains the **SHIFT model**, a spatially resolved techno-economic optimization of global iron and steel supply chains under decarbonization.
It explores how hydrogen-based direct reduced iron (DRI) production and hot-briquetted iron (HBI) trade can shift value creation to regions with renewable energy and capital availability.

The model identifies cost-optimal configurations for mining, hydrogen production, DRI processing, and HBI trade, using a two-stage optimization pipeline and open energy system libraries.

## Introduction SHIFT

SHIFT evaluates global supply and trade of low-carbon iron and steel at high spatial resolution.
The model quantifies where to produce, where to ship, and how to meet demand cost-effectively under decarbonization constraints.

Key features:
- 🚀 two-stage process: greenfield supply curves + cross-region LP trade
- 🌍 spatial renewable potentials: PyPSA-Earth wind/solar CF distributions
- 💰 integrated LCOX: region-level cost curves for H2, DRI, HBI
- 🛳️ global trade dispatch: route costs, flows, nodal prices, utilization
- ⚙️ configurable automation: regions, technologies, scenarios via YAML + Snakemake rules

## Quick installation (PIXI)

Clone the repository:

```sh
git clone https://github.com/energyLS/shift.git
cd shift
```

Install dependencies with pixi:

```sh
pixi install
```

Activate the environment:

```sh
pixi shell
```

(Optional) Install pre-commit hooks:

```sh
pre-commit install
```


For details, see https://pixi.prefix.dev/latest/ (or your local PIXI docs).

## Run (core workflow)

In the workspace root:

```sh
pixi run snakemake model_trade_all
```

To collect all figures (under development):

```sh
pixi run snakemake collect_figures
```


## Workflow overview

### Configuration & Scenario Setup (implicit)

Before execution, Snakemake reads:
- **Global settings**: `config/config.yaml` (regions, cost years, solver options, enable flags)
- **Scenario matrix**: `config/trade_scenarios.csv` (rows = distinct trade scenarios)

These expand into a deterministic wildcard space (cost_year, region, product, scenario) that drives all downstream rule creation. This bootstrap is handled automatically by Snakemake; no user action required.

### Step 0: Renewable potentials (pre-computed inputs)

In this stage we generate the supply-side resource backbone.
Renewable capacity-factor series and maximum deployable potentials are pre-computed externally using PyPSA-Earth's `build_renewable_profiles` rule and stored in `data/renewable_profiles/`.
This stage is not part of the current Snakefile. SHIFT consumes pre-computed .nc datasets to avoid the long runtime of full GIS processing.

> **For new users:** No action needed. Renewable data files are provided in the repository.

### Step 1: Greenfield supply curve generation (PyPSA)

With renewable profiles and [techno-economic assumptions](https://github.com/PyPSA/technology-data) in place, SHIFT builds regional PyPSA optimization models to size generation, storage, and process assets.
It evaluates each candidate plant (H2 electrolyser, DRI furnace, HBI plant, steel mills) across resource quality and cost parameters to produce levelized cost curves (LCOX) as a function of capacity.
The result is a fleet of supply curve elements (capacity buckets with marginal costs and metadata) for H2, DRI, HBI, and steel by region.

Step 1 is a multi-part stage: load techno-economic data and build regional cost baselines, prepare renewable candidate sets per region, solve optimization problems at discrete demand levels, and consolidate results into piecewise supply curves. Each stage depends on the prior; files are persisted between steps to support reproducibility and debugging.
The output is a harmonized set of supply curves used as input for the trade stage.

### Step 2: Global trade optimization (LP)

This stage takes regional supply curves and demand obligations, then runs a linear program over the regional network.
It includes transport cost matrices, ore production constraints, and market compatibility.
The solver decides how much each region should produce versus import/export, by product and route.

The trade solution yields detailed outputs: regional production volume and shipped quantities.
It can also be reconciled with scenarios for demand, policy constraints, and infrastructure availability.


## Acknowledgements

Thanks to:
- Oda Agdal and her Master's Thesis on the [Investigation of Future Global Trade of Hydrogen from Renewable Energy Sources](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3031513)
- Johannes Hampp and [TRACE](https://github.com/euronion/trace)


## Licence

This repository is licensed under the MIT License. See `LICENCE` for details.
