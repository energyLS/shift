# SHIFT – Steel & Hydrogen Integrated Freight Trade

This repository contains the **SHIFT model**, a spatially resolved techno-economic optimization of global iron and steel supply chains under decarbonization. It explores how hydrogen-based direct reduced iron (DRI) production and hot-briquetted iron (HBI) trade can shift value creation to regions with renewable energy, infrastructure, and capital availability.

The model identifies cost-optimal configurations for mining, hydrogen production, DRI processing, and HBI trade, using a two-stage optimization pipeline and open energy system libraries.

## Introduction SHIFT

SHIFT evaluates global supply and trade of low-carbon iron and steel at high spatial resolution. The model quantifies where to produce, where to ship, and how to meet demand cost-effectively under decarbonization constraints.

Key features:
- 🚀 two-stage process: greenfield supply curves + cross-region LP trade
- 🌍 spatial renewable potentials: PyPSA-Earth wind/solar CF distributions 
- 💰 integrated LCOX: region-level cost curves for H2, DRI, HBI
- 🛳️ global trade dispatch: route costs, flows, nodal prices, utilization
- ⚙️ configurable automation: regions, technologies, scenarios via YAML + Snakemake rules

## Quick installation (PIXI)

```sh
git clone https://github.com/energyLS/shift.git && cd shift
python -m pip install --upgrade pip pixi
pixi install
```

For details, see https://pixi.prefix.dev/latest/ (or your local PIXI docs).

## Run (core workflow)

In the workspace root:

```sh
cd workflow
pixi run snakemake -call model_trade_all
```

To collect all figures (under development):

```sh
pixi run snakemake -call collect_figures
```


## Workflow overview

### Step 0: Renewable potentials (Atlite + GIS)

In this stage we generate the supply-side resource backbone. PyPSA-Earth assembles spatial inputs (country polygons, exclusion masks, weather datasets) and computes hourly capacity-factor series and maximum deployable potentials for wind and solar in each region. The workflow uses the [`build_renewable_profiles`](https://pypsa-earth.readthedocs.io/en/latest/user-guide/rules-reference/populate/build-renewable-profiles/) Snakefile rule, and it can produce .nc outputs for per-region, per-technology capacity factor distributions and installable potentials.

> **Note:** SHIFT may consume precomputed Step 0 datasets to avoid the long runtime of full GIS processing; this is the recommended default for day-to-day scenario work.
>
### Step 1: Greenfield supply curve generation (PyPSA)

With renewable profiles and [techno-economic assumptions](https://github.com/PyPSA/technology-data) in place, SHIFT builds regional PyPSA optimization models to size generation, storage, and process assets. It evaluates each candidate plant (H2 electrolyser, DRI furnace, HBI plant, steel mills) across resource quality and cost parameters to produce levelized cost curves (LCOX) as a function of capacity. The result is a fleet of supply curve elements (capacity buckets with marginal costs and metadata) for H2, DRI, HBI, and steel by region.

Each greenfield run schemes the spot around: location selection, renewable share, process stack, cost adders, and available build option integration. The output is a harmonized set of offer curves used as input for the trade stage.

### Step 2: Global trade optimization (LP)

This stage takes regional supply curves and demand obligations, then runs a linear program over the regional network. It includes transport cost matrices, ore production constraints, and market compatibility. The solver decides how much each region should produce versus import/export, by product and route.

The trade solution yields detailed outputs: regional production volume and shipped quantities. It can also be reconciled with scenarios for demand, policy constraints, and infrastructure availability.


## Acknowledgements

Thanks to:
- Oda Agdal and her Master's Thesis on the [Investigation of Future Global Trade of Hydrogen from Renewable Energy Sources](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3031513)
- TRACE?
- PYPSA-earth?


## Licence

This repository is licensed under the MIT License. See `LICENCE` for details.

