# SHIFT – Steel & Hydrogen Integrated Freight Trade

This repository contains the **SHIFT model**, a spatially resolved techno-economic optimization of global iron and steel supply chains under decarbonization. It explores how hydrogen-based direct reduced iron (DRI) production and hot-briquetted iron (HBI) trade can shift value creation to regions with renewable energy, infrastructure, and capital availability.

The model identifies cost-optimal configurations for mining, hydrogen production, DRI processing, and steel trade, using the PyPSA framework.

---

## Download, Install, and Run

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

