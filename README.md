# SHIFT – Steel & Hydrogen Integrated Freight Trade

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21875103.svg)](https://doi.org/10.5281/zenodo.21875103)

**SHIFT** is a spatially resolved techno-economic optimization of global iron
and steel supply chains under decarbonization. It explores how
hydrogen-based direct reduced iron (DRI) production and hot-briquetted iron
(HBI) trade can shift value creation to regions with renewable energy and
capital availability, using a two-stage pipeline (greenfield supply curves,
then cross-region LP trade) built on [PyPSA](https://pypsa.org/) and
orchestrated with [Snakemake](https://snakemake.readthedocs.io/).

## Quick start – installation & execution

SHIFT requires Python 3.10+ and uses [pixi](https://pixi.sh/latest/installation/) to manage the environment. Install pixi before you continue.

```sh
git clone https://github.com/energyLS/shift.git
cd shift
pixi install
pixi run snakemake model_trade_all
```

Approximate runtimes:

- Installation (`pixi install`): ~5 minutes
- Demo model (this branch): ~1 hour

By default, the model solves with [HiGHS](https://highs.dev/), a license-free
solver. If you have a [Gurobi](https://www.gurobi.com/) license, set
`solver.name: gurobi` in `config/config.yaml` for faster solves.

> [!NOTE]
> This is the `demo` branch: it uses fewer regions and quantities so it
> runs much faster. For the full-resolution model (~5 hours), see the
> [`main` branch](https://github.com/energyLS/shift/tree/main).

## Key results

`model_trade_all` populates `results/<scenario>/` for each configured trade
scenario with:

- `result.csv` – regional production volumes and shipped quantities
- `map_ironore.pdf`, `map_hbi.pdf`, `map_steel.pdf` – trade-flow maps by product
- `network.nc` – the full PyPSA network, for further analysis

## Documentation

Full documentation - installation details, solver setup, the workflow
stages, config flags, and a module-by-module reference - lives in `docs/`
and is built with mkdocs:

```sh
pixi run -e docs mkdocs serve
```

then open <http://127.0.0.1:8000>.

## Acknowledgements

Thanks to:
- Oda Agdal and her Master's Thesis on the [Investigation of Future Global Trade of Hydrogen from Renewable Energy Sources](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3031513)
- Johannes Hampp and [TRACE](https://github.com/euronion/trace)


## Licence

This repository is licensed under the MIT License. See `LICENCE` for details.
