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


Run the whole workflow using `snakemake`:

```sh
snakemake -call collect_figures
```


## Licence

This repository is licensed under the MIT License. See `LICENCE` for details.


## Acknowledgements

Thanks to:

* 