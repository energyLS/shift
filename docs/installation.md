# Installation & running the pipeline

## Install dependencies with pixi

```sh
git clone https://github.com/energyLS/shift.git
cd shift
pixi install
pixi shell
```

(Optional) install pre-commit hooks:

```sh
pre-commit install
```

For details on pixi itself, see <https://pixi.prefix.dev/latest/>.

## Solvers

The pipeline solves PyPSA/linopy optimization problems and supports two
solver backends, selected via `config/config.yaml`'s `solver.name` (or
overridden with the `SHIFT_SOLVER` / `SHIFT_SOLVER_OPTIONS` environment
variables):

- **Gurobi** (`solver.name: gurobi`) — the default. Requires a Gurobi
  license (e.g. a `gurobi.lic` file); `gurobipy` is installed automatically.
- **HiGHS** (`solver.name: highs`, or `SHIFT_SOLVER=highs`) — a free
  fallback that needs no license. This is what CI uses, since CI runners
  don't have a Gurobi license.

Only these two are actively configured (`config/config.yaml`'s
`solver_options` only defines `gurobi-default` and `highs-default`
presets). Other solver names some dependencies could technically support
are not wired up with tuned options here.

## Run the core workflow

From the repository root:

```sh
pixi run snakemake model_trade_all
```

On Windows, Snakemake's default job scheduler (an ILP solver via PuLP +
CBC) can fail with a `PulpSolverError` caused by a `-threads` flag
mismatch with the bundled `cbc.exe`. If you hit that, add
`--scheduler greedy`:

```sh
pixi run snakemake --scheduler greedy model_trade_all
```

This only changes how Snakemake orders/parallelizes ready jobs, not what
gets computed.

`collect_figures` does not currently work - its `input:` block references
wildcards (`wacc`, `scenario`, `sort`, `demand`) that are never bound to
concrete values, so Snakemake fails immediately with a `WildcardError`
before building the DAG.
