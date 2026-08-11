# SHIFT – Steel & Hydrogen Integrated Freight Trade

**SHIFT** is a spatially resolved techno-economic optimization of global iron
and steel supply chains under decarbonization. It explores how
hydrogen-based direct reduced iron (DRI) production and hot-briquetted iron
(HBI) trade can shift value creation to regions with renewable energy and
capital availability, using a two-stage pipeline (greenfield supply curves,
then cross-region LP trade) built on [PyPSA](https://pypsa.org/) and
orchestrated with [Snakemake](https://snakemake.readthedocs.io/).

## Quick start

```sh
git clone https://github.com/energyLS/shift.git
cd shift
pixi install
pixi shell
pixi run snakemake model_trade_all
```

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
