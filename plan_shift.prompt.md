
# SHIFT Workflow: High-Level Plan (Steps 0, 1, 2)

## TL;DR

Three-stage workflow: (0) Pre-compute renewable potentials externally → (1) **Constrained regional steel supply curve optimization** under discrete renewable utilization fractions, with local demand as priority load → (2) Global trade LP for **complete supply chain optimization (ore sourcing + HBI/ore trading)** to minimize system cost.

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

## Step 1: Constrained Steel LCO Optimization (PyPSA per Region)

**Purpose:** Generate **LCOSTEEL curves** under discrete renewable potential constraints (1%, 10%, 50%), with local electricity demand pre-served by the most efficient renewable potentials.

**Key Innovation:**
1. Local demand satisfied FIRST using most efficient (highest CF) renewable resources → blocked from steel supply chain
2. Remaining potential available for steel production (H₂ → DRI → HBI → EAF → Steel)
3. Explicit renewable scarcity constraint per region

**Policy Assumptions:**
- Local electricity demand is exogenous infrastructure (non-negotiable baseline; must be served first)
- `renewable_utilization_fractions` apply **only** to remaining capacity after local demand is allocated
- All regions assumed capable of meeting local demand from local renewables (or with grid imports)
- Steel production cost (LCOSTEEL) scales inversely with utilization fraction

### Inputs

1. **Renewable Potentials** (from Step 0)
   - Annual CF and max capacity [GW] per region, technology
   - Stratified by efficiency class (to identify "most efficient" potentials)

2. **Local Electricity Demand (Fixed, Exogenous, Priority)**
   - Regional non-steel electricity demand: `data/un_enerdata_demand_2050_final.csv` → `el_demand [TWh/year]`
   - **Key:** Served FIRST with highest-CF renewable resources; these are blocked from steel production

3. **Steel Demand (Fixed, Exogenous)**
   - Regional steel demand: `resources/steel_production_clustered.csv` [Mt/year]
   - Determines EAF electricity requirement: `EAF_el_demand = steel_demand_Mt × 5.25 [MWh/year]`

4. **Renewable Utilization Constraints**
   - `demand_factors: [0.01, 0.10, 0.50]` – Fraction of **remaining** potential available for steel production
   - After local demand consumes its most-efficient share, constrain H₂ electrolyser capacity to this fraction of what's left

5. **Tech-Economic Parameters** (`config/config.yaml`)
   - Electrolyser efficiency: 75%
   - DRI efficiency: 90%
   - EAF efficiency: 95%
   - Interest rate, CAPEX/OPEX from technology-data v0.12.0

### Process (Per Region, Per Demand-Factor)

```
For region ∈ X_regions:
  
  # STAGE 1: Allocate renewables to local demand (priority)
  # ─────────────────────────────────────────────────────
  Load_local = regional_el_demand  [MWh/year]
  
  # Identify most efficient renewables (highest CF classes)
  # Allocate enough capacity to serve Load_local completely
  Renewables_for_local = select_highest_cf(
    capacity_needed = Load_local / (CF × 8760),
    pool = all_renewable_classes
  )
  
  # Block these from steel supply chain
  Remaining_renewable_potential = Total_capacity - Renewables_for_local
  
  # STAGE 2: Constrain steel supply chain to fraction of remaining potential
  # ──────────────────────────────────────────────────────────────────────
  For demand_factor ∈ [0.01, 0.10, 0.50]:
    
    # Available capacity for steel (H₂ + EAF electricity)
    Available_for_steel = Remaining_renewable_potential × demand_factor
    
    # Build PyPSA network
    # Electricity Bus:
    #   ├─ Generators: Remaining renewables
    #   │   (capacity = Available_for_steel)
    #   ├─ Storage: Battery
    #   ├─ Fixed Load: EAF (EAF_el_demand / 8760 [MW])
    #   │   (competing with electrolyser for same capacity)
    #   └─ Electrolyser → H₂
    #       (flex load, variable output)
    #
    # Objective: minimize CAPEX + OPEX to produce maximum steel
    #   = maximize H₂ production (given EAF demand must be met)
    #
    # Extract cost: LCOSTEEL = total_cost / (steel_demand_Mt)
    
    supply_curve_point = (demand_factor, LCOSTEEL_EUR_per_t)
```

### Outputs (Per Region)

- **Steel Supply Curve (CSV):** `resources/supply_curves/cost_year~{year}/{region}_lcosteel.csv`
  ```
  renewable_utilization_fraction | Steel_demand_Mt | LCOSTEEL_EUR_per_t | H2_prod_MWh | EAF_el_MWh
  0.01                            | steel_demand    | 850                 | 5000        | 41250
  0.10                            | steel_demand    | 420                 | 50000       | 41250
  0.50                            | steel_demand    | 185                 | 250000      | 41250
  ```
  - Each row = discrete renewable utilization scenario
  - EAF electricity is constant (fixed load)
  - H₂ production scales with available renewable capacity

### Key Design Features

- **Local demand prioritized:** Served with most-efficient renewables; no cost optimization for local supply (exogenous)
- **Steel supply chain constrained:** Uses remaining, lower-efficiency potentials
- **Explicit renewable scarcity:** Demand factors (1%, 10%, 50%) directly represent available capacity for steel production
- **Direct product output:** LCOSTEEL generated in Step 1 (no intermediate H₂/HBI curves needed)
- **Realistic resource allocation:** Local demand acts as baseline load with infrastructure priority
- **Iron ore cost = 0:** Not included until Step 2 (avoid circular dependency)

---

## Step 2: Global Supply Chain Optimization (LP)

**Purpose:** Given regional LCOSTEEL curves, optimize global production mix + trade (ore, HBI) to minimize total system cost. **Iron ore cost integration occurs here.**

### Inputs

1. **Steel Supply Curves** (from Step 1)
   - `resources/supply_curves/cost_year~{year}/{region}_lcosteel.csv` for all regions
   - Discrete points: renewable utilization fractions (1%, 10%, 50%) with corresponding LCOSTEEL
   - Iron ore cost NOT included; will be added in LP optimization

2. **Regional Demands (Fixed, Exogenous)**
   - Steel demand: `resources/steel_production_clustered.csv` [Mt/year]

3. **Ore Production Potentials & Cost**
   - Baseline: `resources/ironore_production_clustered.csv` [Mt_ore/year per region]
   - Limit: `production × 1.2` allowance
   - ***Cost: `iron_ore.marginal_cost` [EUR/t_ore]*** (integrated in Step 2 LP)

4. **Transport Costs**
   - Distance matrix: `data/transport_costs/{transport_cost}.csv` [km]
   - Unit cost: `0.005` [€/t·km]

5. **Tech-Economic Parameters** (`config/config.yaml`)
   - Material balance: `ore_to_steel_ratio` = 1.59 [t_ore / t_steel]
   - Other metallurgical parameters from technology-data

6. **Regional Locations** (for distance calculation)
   - `data/bus_locations.csv`: `region_name, lat, long`

7. **Trade Scenarios**
   - `config/trade_scenarios.csv`: scenario modifiers

### Process

**Discrete Supply Curve Integration:**
```
For each region & discrete renewable utilization point:
  # Assemble final cost = LCOSTEEL from Step 1 + ore cost
  Final_cost[region][utilization_point] = LCOSTEEL[region][utilization_point]
                                         + Ore_cost_per_t × (ore_requirement_per_t_steel)
                                         + transport_ore_cost[route]
```

**Metallurgical Chain (inline in LP—inline in LP, transparent):**
```
For each region & production decision:
  H₂_input [MWh]  
    →[Electrolyser 75%]→ H₂_output [MWh]
    →[DRI 90%]→ DRI_output [t DRI]  
    →[stoichiometry]→ HBI_production [t_HBI]  
    →[EAF + material balance]→ Steel_production [t]

Material balance constraints (explicit):
  Ore_extracted [t_ore] × (1 / 1.59) = HBI_available [t_HBI]
  HBI_production [t_HBI] → Steel_production [t_steel]
```

**Decision Variables:**
- `steel_prod[region][utilization_point]` [Mt]: Steel production level (selects discrete utilization point on LCOSTEEL curve)
- `ore_prod[region]` [t_ore]: Iron ore extraction per region
- `hbi_trade[region_from][region_to]` [t]: HBI shipments between regions
- `ore_trade[region_from][region_to]` [t]: Ore shipments between regions

**Objective Function:**
```
minimize Σ_region Σ_utilization( LCOSTEEL_final[utilization] × steel_prod )
       + Σ_region( ore_extraction_cost × ore_prod )           [***INTEGRATED HERE***]
       + Σ_routes( transport_cost_ore × ore_trade )           [***INTEGRATED HERE***]
       + Σ_routes( transport_cost_hbi × hbi_trade )

where LCOSTEEL_final(utilization) = LCOSTEEL(utilization) + ore_cost_per_t_steel + ore_transport
```

**Constraints:**
- Material balance (H₂ → steel): cascade through DRI + EAF, accounting for losses
- Material balance (ore → HBI): ore_extracted / ore_to_hbi_ratio = HBI_production [t]
- Regional balance: local_steel_demand ≤ local_production + net_imports
- Supply limits: ore_prod ≤ regional_potential × 1.2
- Trade conservation: sum(ore_trade) balanced per region
- Non-negativity

### Outputs

- **Trade Solution (CSV):** `results/cost_year~{year}/trade_{scenario}.csv`
  ```
  region           | H2_prod_MWh | DRI_output_t | HBI_prod_t | HBI_import_t | HBI_export_t | Ore_prod_t | Ore_import_t | Ore_export_t | Steel_prod_t | Total_cost_EUR
  Europe           | 50000       | 5500        | 5000       | 200          | 0            | 2000       | 800          | 0            | 3150         | 1.2e8
  N_W_Africa       | 30000       | 3300        | 3000       | 0            | 2800         | 5000       | 0            | 3200         | 1890         | 8.5e7
  ...
  ```

- **Cost Breakdown:** Detailed disaggregation of system cost by component:
  - Electricity (CAPEX/OPEX renewables)
  - Conversion (DRI, EAF CAPEX)
  - Iron ore (extraction + transport)
  - Trade (HBI + ore shipping)

- **Trade Flows & Sensitivity:** Regional production, trade routes, marginal costs, shadow prices

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
┌──────────────────────────────────────────┐
│ Step 0: Renewables (External)            │
│ Output: CF + max_GW per region           │
│         (stratified by efficiency class)  │
└────────────┬─────────────────────────────┘
             │
             ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 1: Constrained Steel LCO Optimization (PyPSA)           │
│                                                              │
│ Stage 1A: Local Demand (Priority)                           │
│  - Served FIRST using most-efficient renewables             │
│  - No cost optimization (exogenous infrastructure)           │
│  - Potentials blocked from steel supply chain               │
│                                                              │
│ Stage 1B: Steel Supply Chain (Constrained)                  │
│  - Renewable generators [capacity × utilization_factor]     │
│  - Battery storage, EAF load, Electrolyser                  │
│  - Utilization factors: 1%, 10%, 50%                        │
│  - Objective: Minimize LCOSTEEL under capacity constraint   │
│                                                              │
│ Output: LCOSTEEL curves (discrete utilization points)        │
│  └─ {region}_lcosteel.csv                                   │
│  └─ Iron ore cost = 0 (deferred to Step 2)                  │
└────────────┬─────────────────────────────────────────────────┘
             │
             ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 2: Global Trade Optimization (LP)                       │
│                                                              │
│ Inputs: LCOSTEEL curves + ore potentials + distances         │
│                                                              │
│ Decisions:                                                   │
│  - Regional steel production level (utilization point)       │
│  - Ore extraction & sourcing [Cost integrated]               │
│  - HBI & ore trade between regions                          │
│                                                              │
│ Output: Optimal production mix, trade flows                  │
│         Minimize total cost (renewables + ore + transport)   │
└──────────────────────────────────────────────────────────────┘
```

---

## Key Design Decisions

1. **Step 1: Constrained Steel LCO (Core Refactor)**
   - **Benefit:** Direct optimization of final product (steel) under renewable scarcity
   - **Local demand priority:** Served first with most-efficient potentials; realistic infrastructure constraint
   - **Renewable utilization explicit:** Demand factors (1%, 10%, 50%) directly represent available capacity for steel supply chain
   - **Advantage:** Simpler conceptually; transparent resource allocation; direct product output (LCOSTEEL)

2. **Local Demand Pre-Allocation (NO Cost Optimization)**
   - **Rationale:** Represents baseline infrastructure demand; exogenous constraint
   - **Implementation:** Allocate most-efficient renewables to local load; block these potentials
   - **Outcome:** Steel supply chain competes for lower-tier renewable resources (realistic)

3. **Iron Ore Cost = 0 in Step 1 (Retained)**
   - **Rationale:** Avoids circular dependency (ore cost ↔ steel cost ↔ ore sourcing)
   - **Integration:** Ore cost added ONLY in Step 2 LP
   - **Outcome:** Step 1 provides clean LCOSTEEL vs. renewable constraint curves

4. **Direct Product Output (Steel)**
   - **Benefit:** LCOSTEEL generated directly in Step 1; no intermediate curves needed
   - **Efficiency:** Single PyPSA optimization produces final supply curve
   - **Outcome:** Clean, direct data flow from Step 1 → Step 2

5. **Electricity Competition Implicit (Sequential Allocation)**
   - **In Step 1:** Local demand + steel production compete for remaining renewables; local demand wins (priority)
   - **In Step 2:** No further electricity competition; EAF supply is implicit in LCOSTEEL curves
   - **Result:** Clear resource hierarchy; no ambiguous "competing loads" concept

6. **Trade Scope: HBI + Ore (Unchanged Logic)**
   - **H₂:** Local production only (high transport cost)
   - **HBI:** Tradeable intermediate (lower shipping cost than final product)
   - **Ore:** Tradeable raw material (regional potentials differ significantly)
   - **Steel:** Manufactured locally from HBI + ore (final product has lowest transport intensity)

---

## Key Parameters (config.yaml)

| Parameter | Value | Unit | Notes |
|-----------|-------|------|-------|
| `renewable_utilization_fractions` | [0.01, 0.10, 0.50] | fraction | Discrete scenarios for Step 1 renewable utilization (after local demand served) |
| `electrolyser_efficiency` | 0.75 | — | H₂ output / electricity input |
| `DRI_efficiency` | 0.90 | — | Direct reduction iron yield |
| `electricity_steel_ratio` | 5.25 | MWh/t | EAF electricity per tonne steel (fixed load in Step 1) |
| `ore_to_steel_ratio` | 1.59 | t_ore/t_steel | Ore requirement per steel output (Step 2) |
| `shipping_cost` | 0.005 | €/t·km | Transport cost for ore/HBI (Step 2 only) |
| `iron_ore.marginal_cost` | 97.7 | €/t_ore | **Added in Step 2 LP only** |
| `iron_ore.potential_allowance` | 1.2 | factor | Upside limit on regional ore extraction |

---

## Summary: What Changed & Why

| Aspect | Original Plan | New Approach | Reason |
|--------|---------------|--------------|--------|
| **Step 1 goal** | Generate generic electricity curves | Optimize LCOSTEEL under renewable constraint | Direct product; explicit scarcity |
| **Local demand** | Competing load (same bus as H₂) | Priority load (served first, best potentials) | Realistic infrastructure hierarchy |
| **Renewable allocation** | All available for H₂ + EAF competition | Two-stage: local demand first, then steel | Explicit resource sequencing |
| **Supply curve output** | Electricity curve (foundation) | LCOSTEEL curve (final product) | Simpler, more direct |
| **Demand factors** | 0.1% - 70% (H₂ utilization) | 1%, 10%, 50% (renewable availability *after* local demand) | Constrained scarcity scenario |
| **Cost accumulation** | Multi-stage generic chains | Implicit in single PyPSA solve | No intermediate abstractions needed |
| **Iron ore cost** | = 0 (config toggle) | = 0 (explicit design principle) | Avoids circular dependency |

---

## Implementation Roadmap

### Phase 1: Refactor to Constrained Steel LCO
1. Modify Step 1 PyPSA model to implement two-stage renewable allocation:
   - Stage 1A: Pre-serve local demand with highest-CF renewables (no optimization)
   - Stage 1B: Optimize LCOSTEEL with remaining capacity under utilization constraint
2. Update Snakemake rule `model_lcox` to parameterize renewable utilization factor
3. Generate LCOSTEEL supply curves at discrete utilization points (1%, 10%, 50%)
4. Remove unnecessary `model_lcoh.py` or consolidate into single model
5. Validate outputs match mission possible steel / current LCOX calculations

### Phase 2: Enhanced Analysis (Optional)
1. Add cost component breakdowns to LCOSTEEL supply curves (electricity, conversion, ore (=0 at this stage))
2. Document sensitivity to renewable efficiency class selection
3. Explore alternative local demand scenarios if needed

### Phase 3: Validate & Extend
1. Cross-validate LCOSTEEL against existing TRACE model results
2. Test sensitivity to renewable efficiency class selection
3. Document all assumptions and model boundaries