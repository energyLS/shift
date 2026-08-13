# Workflow overview

### Configuration & scenario setup (implicit)

Before execution, Snakemake reads:

- **Global settings**: `config/config.yaml` (regions, cost years, solver options, enable flags)
- **Scenario matrix**: `config/trade_scenarios.csv` (rows = distinct trade scenarios)

These expand into a deterministic wildcard space (cost_year, region,
product, scenario) that drives all downstream rule creation. This
bootstrap is handled automatically by Snakemake; no user action required.

### The `enable` flags

`config/config.yaml`'s `enable` block gates three of the most expensive
rules in the pipeline, and ships **`True`** by default so a fresh clone
runs end-to-end with no manual edits:

```yaml
enable:
  run_supply_chain: True   # gates rule calculate_regional_lcox
  run_supply_curve: True   # gates rule create_supply_curve
  cluster_renewables: True # gates rule cluster_renewables
```

These aren't ordinary "skip if already done" toggles - each one wraps an
entire `rule` definition in an `if config["enable"].get(...)` in the
`.smk` files, so when a flag is `False`, Snakemake doesn't just skip
re-running that rule, **the rule doesn't exist in the DAG at all**:

- `cluster_renewables: False` removes the `cluster_renewables` rule, so
  `resources/renewables_clustered.nc` must already exist on disk.
- `run_supply_chain: False` removes `calculate_regional_lcox`, so the
  per-region/product/demand-level LCOX result CSVs it would produce
  (`resources/lco-{product}/.../results_{demand}.csv`) must already exist.
- `run_supply_curve: False` removes `create_supply_curve`, so the
  combined supply-curve CSVs/PDFs
  (`resources/supply_curves/.../{region}_marginal_cost_{product}.csv`)
  must already exist.

`resources/` is gitignored - nothing under it ships in the repo - so all
three flags must be `True` the first time you run against a given config
(regions, cost year, scenarios). Once the resources exist on disk, you
can set them back to `False` to skip re-deriving them on subsequent runs
- useful because renewable clustering alone takes on the order of 10+
minutes, and LCOX/supply-curve generation scales with the number of
configured regions.

The trap: if you change something the cached resources depend on (most
obviously `regions`, but also `cost_year` or clustering parameters) while
leaving a flag `False`, Snakemake won't silently regenerate the stale
file for you - since the rule doesn't exist, you'll either get a hard
`MissingInputException`/`No rule to produce ...` error, or worse, a run
that silently reuses resources computed for the *previous* region set.
When in doubt (e.g. after editing `regions:`), set all three back to
`True` for one run.

### Step 0: Renewable potentials

Renewable capacity-factor series and maximum deployable potentials are
retrieved from a pre-computed Zenodo archive (see the `retrieve_data`
rule) rather than generated locally, to avoid the long runtime of full
GIS processing via PyPSA-Earth's `build_renewable_profiles`.

### Step 1: Greenfield supply curve generation (PyPSA)

With renewable profiles and [techno-economic assumptions](https://github.com/PyPSA/technology-data)
in place, SHIFT builds regional PyPSA optimization models to size
generation, storage, and process assets. It evaluates each candidate
plant (H2 electrolyser, DRI furnace, HBI plant, steel mills) across
resource quality and cost parameters to produce levelized cost curves
(LCOX) as a function of capacity. The result is a fleet of supply curve
elements (capacity buckets with marginal costs and metadata) for H2, DRI,
HBI, and steel by region.

Step 1 is a multi-part stage: load techno-economic data and build
regional cost baselines, prepare renewable candidate sets per region,
solve optimization problems at discrete demand levels, and consolidate
results into piecewise supply curves. Each stage depends on the prior;
files are persisted between steps to support reproducibility and
debugging. See [`calculate_lcox`](modules.md#calculate_lcox) and
[`create_supply_curve`](modules.md#create_supply_curve).

### Step 2: Global trade optimization (LP)

This stage takes regional supply curves and demand obligations, then runs
a linear program over the regional network. It includes transport cost
matrices, ore production constraints, and market compatibility. The
solver decides how much each region should produce versus import/export,
by product and route.

The trade solution yields detailed outputs: regional production volume
and shipped quantities. It can also be reconciled with scenarios for
demand, policy constraints, and infrastructure availability. See
[`model_trade`](modules.md#model_trade).
