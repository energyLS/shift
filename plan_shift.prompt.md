
# SHIFT Workflow: High-Level Plan (Steps 0, 1, 2)

## TL;DR

Three-stage workflow: (0) Pre-compute renewable potentials externally → (1) Regional greenfield PyPSA optimization for **hydrogen with EAF + local demand as fixed competing loads**, yielding LCOH supply curves → (2) Global trade LP for **complete supply chain (H₂ → DRI → HBI → EAF → steel) + ore trade**.

---

## Step 0: Renewable Potentials (External, Pre-computed)

**Scope:** Data aggregation; largely handled outside SHIFT via PyPSA-Earth

**Outputs from PyPSA-Earth:**
- Hourly capacity factors and potential power generation (solar, wind_on, wind_off): `profile_solar.nc`, `profile_wind_on.nc`, `profile_wind_off.nc` per region
- Weather year: 2013

**SHIFT Integration Point (Data Aggregation Only):**
- Ingest Step 0 outputs
- Aggregate regional, but keep hourly resolution for supply curve generation

**File Locations:**
- Input: `data/renewable_profiles/profile_{technology}.nc`
- Output (Step 1 input): Regional renewable supply table

---

## Step 1: Greenfield H₂ Supply Curve Generation (PyPSA per Region)

**Purpose:** Generate **LCOH curves** (levelized cost of hydrogen) at each discrete renewable utilization scenario. **EAF and local electricity demand are fixed competing loads on the same electricity bus.**

### Inputs

1. **Renewable Potentials** (from Step 0)
   - Annual CF and max [GW] per region, technology

2. **Demand Data (Fixed, Exogenous)**
   - Regional steel demand: `resources/steel_production_clustered.csv` → compute `EAF_el_demand = steel_demand_Mt × 5.25` [MWh/year]
   - Regional non-steel electricity demand: `data/un_enerdata_demand_2050_final.csv` → `el_demand_TWh [TWh/year]` with `el_share [%]`

3. **Tech-Economic Parameters** (`config/config.yaml`)
   - Electrolyser efficiency: 75%
   - `demand_factors: [0.001, 0.01, ..., 70%]` – Discrete renewable utilization scenarios for H₂
   - Interest rate, CAPEX/OPEX costs from technology-data v0.12.0

### Process (Per Region, Per Demand-Factor)

```
For region ∈ X_regions:
  For demand_factor ∈ demand_factors:
    
    # Calculate fixed electricity loads (non-optimizable)
    EAF_el_demand = steel_demand_Mt × 5.25           [MWh/year]
    Local_el_demand = regional_el_demand × el_share  [MWh/year]
    Total_fixed_load = EAF_el_demand + Local_el_demand
    
    # Build PyPSA network
    # - Generators: PV, Wind_on, Wind_off 
    #   (capacity limited by max_potential × demand_factor)
    # - Storage: Battery,
    # - Loads (Fixed, constant p_set):
    #   → EAF (fixed): p_set = EAF_el_demand / 8760 [MW]
    #   → Local (fixed): p_set = Local_el_demand / 8760 [MW]
    # - Links: Electrolyser (75% efficiency)
    #   → Output: H₂ at demand_factor × max_potential
    # - Solve: minimize CAPEX + OPEX 
    #   (supply EAF + Local + H₂_product simultaneously)
    
    # Extract cost
    supply_curve_point = (demand_factor, LCOH_EUR_per_MWh_H2)
```

### Outputs (Per Region)

- **H₂ Supply Curve (CSV):** `resources/supply_curves/cost_year~2030/{region}_hydrogen.csv`
  ```
  demand_MWh | LCOH_EUR_per_MWh | demand_factor
  1000       | 125              | 0.1
  2000       | 135              | 1.0
  ...
  ```
  - Each row = discrete demand-factor solved
  - Continuous interpolation for Step 2

### Key Design Features

- **Electricity competition explicit:** EAF + local + hydrogen all compete for the same renewable supply
- **Higher EAF demand → higher LCOH:** Renewable capacity must first satisfy EAF + local; remainder available for H₂
- **Output is hydrogen only:** No metallurgical conversions; Step 1 ends at H₂ product
- **Iron ore cost excluded:** Cannot cause circular dependency (ore cost is in Step 2 only)

---

## Step 2: Global Supply Chain Optimization (LP)

**Purpose:** Given H₂ supply curves, optimize complete supply chain (H₂ → HBI → steel) + ore sourcing and trade to minimize global cost.

### Inputs

1. **H₂ Supply Curves** (from Step 1)
   - `resources/supply_curves/cost_year~2030/{region}_hydrogen.csv` for all 15 regions

2. **Regional Demands (Fixed, Exogenous)**
   - Steel demand: `resources/steel_production_clustered.csv` [Mt/year]

3. **Ore Production Potentials**
   - Baseline: `resources/ironore_production_clustered.csv` [Mt_ore/year per region]
   - Limit: `production × 1.2` allowance

4. **Transport Costs**
   - Distance matrix: `data/transport_costs/{transport_cost}.csv` [km]
   - Unit cost: `0.005` [€/t·km]

5. **Tech-Economic Parameters** (`config/config.yaml`)
   - DRI kiln efficiency: 90% (ore → DRI)
   - Other CAPEX/OPEX from technology-data

6. **Regional Locations** (for distance calculation)
   - `data/bus_locations.csv`: `region_name, lat, long`

7. **Trade Scenarios**
   - `config/trade_scenarios.csv`: scenario modifiers

### Process

**Metallurgical Chain (inline in LP):**
```
For each region:
  H₂_input [MWh]  
    →[DRI 90%]→ DRI_output [t_DRI]  
    →[stoichiometry]→ HBI_production [t_HBI]  
    →[material balance]→ consumed for steel production

Material balance constraints:
  Ore_extracted [t_ore] × (1 / 1.59) = HBI_production [t_HBI]
  (where 1.59 = ore_to_steel_ratio)
  HBI_production [t_HBI] → Steel_production [t_steel]

EAF electricity (external, fixed, zero cost):
  EAF_el_supply = steel_demand_Mt × 5.25 [MWh/year]
  (Pre-allocated; NOT optimized in Step 2)
```

**Decision Variables:**
- `h2_prod[region][curve_idx]` [MWh]: H₂ production on curve point
- `ore_prod[region]` [t_ore]: Iron ore extraction per region
- `hbi_trade[region_from][region_to]` [t_HBI]: HBI shipments
- `ore_trade[region_from][region_to]` [t_ore]: Ore shipments

**Objective Function:**
```
minimize Σ_region Σ_curve( LCOH[curve] × h2_prod )
       + Σ_region( DRI_cost_per_t × h2_to_dri[region] )
       + Σ_region( ore_extraction_cost × ore_prod )
       + Σ_route( transport_cost_ore × ore_trade )
       + Σ_route( transport_cost_hbi × hbi_trade )
```

**Constraints:**
- Material balance (H₂ → DRI): h2_prod × efficiency_factor = DRI_output [t]
- Material balance (ore → HBI): ore_prod / 1.59 = HBI_production [t]
- Material balance (HBI demand): total_hbi_available ≥ sum(regional_steel_demand × ore_requirement)
- Supply limits: ore_prod ≤ regional_potential × 1.2
- Non-negativity

### Outputs

- **Trade Solution (CSV):** `results/cost_year~2030/trade_{scenario}.csv`
  ```
  region           | H2_prod_MWh | DRI_output_t | HBI_prod_t | HBI_import_t | HBI_export_t | Ore_prod_t | Ore_import_t | Ore_export_t | Steel_prod_t | Cost_EUR
  Europe           | 50000       | 5500        | 5000       | 200          | 0            | 2000       | 800          | 0            | 3150         | 1.2e8
  N_W_Africa       | 30000       | 3300        | 3000       | 0            | 2800         | 5000       | 0            | 3200         | 1890         | 8.5e7
  ...
  ```

- **Detailed Results:** Marginal costs, shadow prices, trade flows, objective value

- **Visualizations:** Regional production, trade routes, cost breakdown, sensitivity analysis

---

## Critical Data Files (All Confirmed ✓)

| File | Purpose | Source | Status |
|------|---------|--------|--------|
| `resources/steel_production_clustered.csv` | Regional steel demand [Mt/year] | OWID clustered | ✓ |
| `resources/ironore_production_clustered.csv` | Regional iron ore production [Mt_ore/year] | OWID clustered | ✓ |
| `data/un_enerdata_demand_2050_final.csv` | Regional final energy + electricity share [%] | External | ✓ |
| `data/bus_locations.csv` | Regional centroids (lat/long) | Hardcoded | ✓ |
| `data/transport_costs/{transport_cost}.csv` | Distance matrix [km] | Computed | ✓ |
| `config/trade_scenarios.csv` | Trade scenario definitions | Config | ✓ |
| `data/new_renewables/supply_*.nc` | Renewable CF + max potentials | PyPSA-Earth | ✓ |
| `config/config.yaml` | All parameters (costs, ratios, tech-data) | Config | ✓ |

---

## Data Flow Architecture

```
┌──────────────────────────────────┐
│ Step 0: Renewables (External)    │
│ Output: CF + max_GW per region   │
└────────────┬─────────────────────┘
             │
             ↓
┌──────────────────────────────────────────────────┐
│ Step 1: Regional PyPSA (H₂ Only)                 │
│                                                  │
│ Electricity bus:                                 │
│  ├─ Renewables (PV, Wind)                        │
│  ├─ EAF demand [fixed load]                      │
│  ├─ Local demand [fixed load]                    │
│  └─ Electrolyser → H₂ [variable]                 │
│                                                  │
│ Output: LCOH curves {region}_hydrogen.csv        │
└────────────┬─────────────────────────────────────┘
             │
             ↓
┌──────────────────────────────────────────────────┐
│ Step 2: Global LP (Complete Supply Chain)        │
│                                                  │
│ H₂ (from Step 1)                                 │
│  →[DRI 90%]→ HBI                                 │
│  + Ore extraction & trade                        │
│  + HBI trade                                     │
│  →[material balance]→ EAF (fixed)→ Steel         │
│                                                  │
│ Output: Trade solution, visualizations           │
└──────────────────────────────────────────────────┘
```

---

## Key Design Decisions

1. **Step 1 Scope: Electricity → H₂ Only**
   - Benefit: Simpler PyPSA network, shorter solve time
   - Electricity competition explicit: EAF + local are fixed loads
   - LCOH already reflects competition for renewable capacity

2. **Iron Ore Cost in Step 2 (Not Step 1)**
   - Rationale: Avoids circular dependency (ore cost ↔ H₂ cost)
   - Step 1 is purely renewable + H₂ electrolysis
   - Step 2 integrates H₂ curves with metallurgical conversions + ore costs

3. **EAF Electricity Fixed (External)**
   - In Step 1: Acts as fixed competing load (competes with H₂ for renewables)
   - In Step 2: Provided externally (zero cost, fixed quantity = steel_demand × 5.25)
   - Rationale: EAF is exogenous constraint; not optimized

4. **Trade Scope: HBI + Ore (Not H₂ or Steel)**
   - H₂: Local production only (transport cost prohibitive)
   - HBI: Tradeable intermediate (minimizes bulk of final product)
   - Ore: Tradeable raw material (regions have different potentials)
   - Steel: Manufactured locally from HBI + ore

---

## Key Parameters (config.yaml)

| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| `demand_factors` | [0.001, ..., 0.70] | fraction | Discrete scenarios for Step 1 |
| `electrolyser_efficiency` | 0.75 | — | H₂ output / electricity input |
| `DRI_efficiency` | 0.90 | — | Direct reduction iron yield |
| `electricity_steel_ratio` | 5.25 | MWh/t | EAF electricity per tonne steel |
| `ore_to_steel_ratio` | 1.59 | t_ore/t_steel | Ore requirement per steel output |
| `shipping_cost` | 0.005 | €/t·km | Transport cost for ore/HBI |

---

## Next Steps

1. **Detailed Plan 1:** Step 1 Snakemake rules, PyPSA network template, discrete scenario handling
2. **Detailed Plan 2:** Step 2 LP structure with piecewise-linear H₂ cost assembly