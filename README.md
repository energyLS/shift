# SHIFT – Steel & Hydrogen Integrated Freight Trade

This repository contains the **SHIFT model**, a spatially resolved techno-economic optimization of global iron and steel supply chains under decarbonization. It explores how hydrogen-based direct reduced iron (DRI) production and hot-briquetted iron (HBI) trade can shift value creation to regions with renewable energy, infrastructure, and capital availability.

The model identifies cost-optimal configurations for mining, hydrogen production, DRI processing, and steel trade, using the PyPSA framework.

---


## Prerequesite: TRACE model

The SHIFT model integrates the energy supply chain `shipping-steel` from a [fork](https://github.com/fneum/trace/tree/pypsa-eur-sec-imports-atlite) of the [TRACE model](https://github.com/euronion/trace).

Therefore, clone the TRACE fork with `git`:

```sh
git clone https://github.com/fneum/trace.git
```

and switch to the branch `pypsa-eur-sec-imports-atlite` (commit [8bf0571](https://github.com/fneum/trace/commit/8bf057142d4e035926ffb084493462eff64fe188)) and follow these steps:
- delete `escs/shipping-steel/loads.csv`,
- delete `escs/shipping-steel/ships.csv`,
- set `technology_data: "v0.12.0"` in the `config/config.default.yaml` (same as in SHIFT: `config/config.yaml`),
- run `snakemake -c1 resources/networks/default/2030/shipping-steel/DE-DE/network.nc`,
- run `snakemake -c1 resources/networks/default/2050/shipping-steel/DE-DE/network.nc`.

This creates a steel supply chain for 2030 and 2050 without loads and shipping, those parameters will be added later in the SHIFT workflow. The resulting steel model will be stored in `resources/networks/default/2050/shipping-steel/DE-DE/network.nc` and automatically fetched by the SHIFT model.


## Download, Install, and Run the SHIFT model

Clone the repository with `git`:

```sh
git clone https://github.com/energyLS/shift.git
```

Create the environment with `conda`:

```sh
conda env create -f environment.yaml
```

Run the trade model by navigating to the `workflow/` folder via `cd workflow` and then run

```sh
snakemake -call model_trade_all
```

To plot the supply curves subtracted with demand, run

```sh
snakemake -c1 create_all_supply_curves_with_demand
```


*Under development:*

Run the whole workflow using `snakemake`:

```sh
snakemake -call collect_figures
```


## Licence

This repository is licensed under the MIT License. See `LICENCE` for details.


## Acknowledgements

Thanks to:

* Oda Agdal and her Master's Thesis on the [Investigation of Future Global Trade of Hydrogen from Renewable Energy Sources](https://ntnuopen.ntnu.no/ntnu-xmlui/handle/11250/3031513).

