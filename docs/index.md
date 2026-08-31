# SHIFT — Steel & Hydrogen Integrated Freight Trade

SHIFT is a spatially resolved techno-economic optimization of global iron and
steel supply chains under decarbonization. It explores how hydrogen-based
direct reduced iron (DRI) production and hot-briquetted iron (HBI) trade can
shift value creation to regions with renewable energy and capital
availability.

The model identifies cost-optimal configurations for mining, hydrogen
production, DRI processing, and HBI trade, using a two-stage optimization
pipeline built on [PyPSA](https://pypsa.org/) and orchestrated with
[Snakemake](https://snakemake.readthedocs.io/).

Key features:

- **Two-stage process**: greenfield supply curves + cross-region LP trade
- **Spatial renewable potentials**: PyPSA-Earth wind/solar CF distributions
- **Integrated LCOX**: region-level cost curves for H2, DRI, HBI
- **Global trade dispatch**: route costs, flows, nodal prices, utilization
- **Configurable automation**: regions, technologies, scenarios via YAML + Snakemake rules

## Where to go next

- [Installation & running the pipeline](installation.md) — pixi setup, the
  core Snakemake commands, and known platform-specific gotchas (notably a
  Snakemake scheduler issue on Windows).
- [Workflow overview](workflow.md) — what each pipeline stage does.
- [Module reference](modules.md) — auto-generated documentation for each
  script in `workflow/scripts/`, pulled directly from their docstrings.
