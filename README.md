# SHIFT – Steel & Hydrogen Integrated Freight Trade

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENCE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21875103.svg)](https://doi.org/10.5281/zenodo.21875103)

**SHIFT** is a spatially resolved techno-economic optimization of global iron
and steel supply chains under decarbonization. It explores how
hydrogen-based direct reduced iron (DRI) production and hot-briquetted iron
(HBI) trade can shift value creation to regions with renewable energy and
capital availability, using a two-stage pipeline (greenfield supply curves,
then cross-region LP trade) built on [PyPSA](https://pypsa.org/) and
orchestrated with [Snakemake](https://snakemake.readthedocs.io/).

## Quick start – installation & execution

```sh
git clone https://github.com/energyLS/shift.git
cd shift
pixi install
pixi run snakemake model_trade_all
```

Approximate runtimes:

- Installation (`pixi install`): ~5 minutes
- Full model (`model_trade_all`): ~5 hours

> [!TIP]
> For a quick, low-resolution run, check out the
> [`demo` branch](https://github.com/energyLS/shift/tree/demo), which uses
> fewer regions and quantities and completes in ~1 hour.

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
