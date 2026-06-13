"""Helpers for normalizing trade-chain config and deriving stage groups.

The config currently stores a single trade chain with ordered stages keyed by
stage number. This module turns that into a stable, ordered representation and
derives the stage group boundaries implied by tradeable commodities.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from _helpers import setup_logging

logger = setup_logging(__name__)


ENERGY_INPUTS = {"renewable_electricity", "grid_electricity"}
BUS_ALIASES = {
    "H2": "hydrogen",
    "h2": "hydrogen",
    "hydrogen": "hydrogen",
    "renewable_electricity": "renewable_electricity",
    "grid_electricity": "grid_electricity",
}

# Explicit mapping from high-level process identifiers to concrete PyPSA components.
# Keep this mapping authoritative so config `process_label` remains high-level.
TECH_COMPONENT_MAP = [
    {
        "match": ("electro", "electrolyser", "electrolyzer"),
        # Some skeletons name this link `electrolysis` while others use
        # `electrolyzer`/`electrolyser`. Include common variants so slicer
        # keeps the actual link present in the network.
        "links": ("electrolyzer", "electrolysis", "electrolyser"),
        "stores": ("h2_storage",),
        # expected material reactants, energy inputs, and outputs
        "materials": (),
        "energy": ("renewable_electricity",),
        "outputs": ("hydrogen",),
        "buses": ("hydrogen", "renewable_electricity"),
    },
    {
        "match": ("dri", "direct_reduction", "reduction"),
        "links": ("dri",),
        "stores": ("h2_storage", "hbi_storage"),
        "materials": ("iron_ore", "hydrogen"),
        "energy": ("renewable_electricity",),
        "outputs": ("hbi",),
        "buses": (
            "iron_ore",
            "hydrogen",
            "hbi",
            "renewable_electricity",
            "grid_electricity",
        ),
    },
    {
        "match": ("eaf", "electric_arc", "arc_furnace"),
        # Support both `eaf` and `eaf-grid` link namings found in skeletons.
        "links": ("eaf", "eaf-grid", "electric_arc_furnace"),
        "stores": ("steel_storage",),
        "materials": ("hbi",),
        "energy": ("grid_electricity", "renewable_electricity"),
        "outputs": ("steel",),
        "buses": ("hbi", "steel", "grid_electricity", "renewable_electricity"),
    },
]


def _components_for_process_label(
    process_label: str,
) -> Optional[Dict[str, Tuple[str, ...]]]:
    """Return mapped components and expected IO for a given process_label or None if not found."""
    if not process_label:
        return None
    pl = process_label.lower()
    for entry in TECH_COMPONENT_MAP:
        for pat in entry["match"]:
            if pat in pl:
                return {
                    "links": tuple(entry.get("links", ())),
                    "stores": tuple(entry.get("stores", ())),
                    "buses": tuple(entry.get("buses", ())),
                    "materials": tuple(entry.get("materials", ())),
                    "energy": tuple(entry.get("energy", ())),
                    "outputs": tuple(entry.get("outputs", ())),
                }
    return None


def validate_stage_io(stage: Dict, raise_on_mismatch: bool = False) -> bool:
    """Validate that a stage's declared inputs/outputs match the canonical mapping.

    Returns True if validation passes or no mapping exists. If `raise_on_mismatch` is True,
    a ValueError is raised on mismatch; otherwise a warning is returned via logging and False is returned.
    """
    process_label = str(stage.get("process_label", "")).strip()
    if not process_label:
        return True

    comp = _components_for_process_label(process_label)
    if comp is None:
        # No mapping — nothing to validate
        return True

    # Normalize declared inputs/outputs
    declared_materials, declared_energy = split_stage_inputs(stage)
    declared_materials_norm = {_normalize_commodity(m) for m in declared_materials}
    declared_energy_norm = {_normalize_commodity(e) for e in declared_energy}
    declared_output = _normalize_commodity(stage.get("output_commodity", ""))

    expected_materials = {_normalize_commodity(m) for m in comp.get("materials", ())}
    expected_energy = {_normalize_commodity(e) for e in comp.get("energy", ())}
    expected_outputs = {_normalize_commodity(o) for o in comp.get("outputs", ())}

    msgs = []
    # Materials: declared_materials should be a superset of expected_materials or vice versa?
    # We allow declared to be a superset (user may include both H2 and iron_ore), but require at least one overlap
    if expected_materials and declared_materials_norm.isdisjoint(expected_materials):
        msgs.append(
            f"Stage '{process_label}': declared material inputs {declared_materials_norm} do not overlap expected {expected_materials}"
        )

    # Energy: declared energy should overlap expected energy
    if expected_energy and declared_energy_norm.isdisjoint(expected_energy):
        msgs.append(
            f"Stage '{process_label}': declared energy inputs {declared_energy_norm} do not overlap expected {expected_energy}"
        )

    # Output: declared_output should be one of expected outputs
    if expected_outputs and declared_output and declared_output not in expected_outputs:
        msgs.append(
            f"Stage '{process_label}': declared output '{declared_output}' not in expected outputs {expected_outputs}"
        )

    if msgs:
        if raise_on_mismatch:
            raise ValueError("; ".join(msgs))
        for m in msgs:
            logger.warning(m)
        return False

    return True


def iter_trade_chains(config: Dict) -> List[Dict]:
    """Return trade chain definitions as a list."""

    trade_chains = config.get("trade_chains")
    if not trade_chains:
        return []
    if isinstance(trade_chains, dict):
        return [trade_chains]
    return list(trade_chains)


def get_trade_chain(config: Dict) -> Dict:
    """Return the primary trade chain from config."""

    chains = iter_trade_chains(config)
    if not chains:
        return {}
    return chains[0]


def get_ordered_stages(chain: Dict) -> List[Dict]:
    """Return stages ordered by their numeric key or declared order."""

    stages = chain.get("stages", {})
    if isinstance(stages, dict):
        items = sorted(stages.items(), key=lambda item: int(item[0]))
        ordered = []
        for key, stage in items:
            stage_dict = dict(stage)
            stage_dict.setdefault("order", int(key))
            ordered.append(stage_dict)
        return ordered

    ordered = [dict(stage) for stage in stages]
    ordered.sort(key=lambda stage: int(stage.get("order", 0)))
    return ordered


def _as_list(value) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def _normalize_commodity(name: str) -> str:
    return BUS_ALIASES.get(str(name), str(name))


def split_stage_inputs(stage: Dict) -> Tuple[List[str], List[str]]:
    """Split a stage's inputs into material inputs and energy inputs."""

    if "material_inputs" in stage or "energy_inputs" in stage:
        raw_inputs = _as_list(stage.get("material_inputs"))
        energy_inputs = _as_list(stage.get("energy_inputs"))
    elif "input_commodities" in stage:
        raw_inputs = _as_list(stage.get("input_commodities"))
        energy_inputs = _as_list(stage.get("energy_inputs"))
    else:
        raw_inputs = _as_list(stage.get("input_commodity"))
        energy_inputs = []

    materials = []
    energy = list(energy_inputs)

    for input_name in raw_inputs:
        if input_name in ENERGY_INPUTS:
            energy.append(input_name)
        else:
            materials.append(input_name)

    return materials, energy


def get_stage_groups(chain: Dict) -> List[Dict]:
    """Group contiguous stages until a tradeable output or the final product."""

    ordered_stages = get_ordered_stages(chain)
    if not ordered_stages:
        return []

    tradeable = {str(item) for item in chain.get("tradeable_commodities", [])}
    final_product = str(chain.get("final_product", "")).strip()

    groups = []
    current = []
    for stage in ordered_stages:
        current.append(stage)
        output_commodity = str(stage.get("output_commodity", "")).strip()
        if output_commodity in tradeable or output_commodity == final_product:
            groups.append(
                {
                    "label": output_commodity,
                    "stages": list(current),
                }
            )
            current = []

    if current:
        last_output = str(current[-1].get("output_commodity", "")).strip()
        groups.append({"label": last_output, "stages": list(current)})

    return groups


def route_label_for_product(config: Dict, product: str) -> str:
    """Return the stage-group label for a product."""

    chain = get_trade_chain(config)
    for group in get_stage_groups(chain):
        if group["label"] == product:
            return group["label"]
    return product


def derive_supply_curve_products(config: Dict) -> List[str]:
    """Return the externally visible product list for supply curves."""

    products = []
    for chain in iter_trade_chains(config):
        for group in get_stage_groups(chain):
            label = group["label"]
            if label and label not in products:
                products.append(label)
    return products or ["steel"]


def build_product_components(config: Dict, product: str) -> Dict[str, object]:
    """Derive the links, stores, buses, and renewable flag for a product."""

    chain = get_trade_chain(config)
    groups = get_stage_groups(chain)

    target_group: Optional[Dict] = None
    for group in groups:
        if group["label"] == product:
            target_group = group
            break

    if target_group is None:
        raise ValueError(f"Product '{product}' not found in configured stage groups")

    links = set()
    stores = set()
    buses = set()
    has_renewables = False

    for stage in target_group["stages"]:
        process_label = str(stage.get("process_label", "")).strip()
        materials, energy = split_stage_inputs(stage)

        # Add explicit buses derived from stage inputs
        for material in materials:
            buses.add(_normalize_commodity(material))
        for energy_input in energy:
            norm = _normalize_commodity(energy_input)
            buses.add(norm)
            if energy_input == "renewable_electricity":
                has_renewables = True

        output_commodity = _normalize_commodity(stage.get("output_commodity", ""))
        if output_commodity:
            buses.add(output_commodity)

        # Use explicit mapping from process_label -> concrete components
        comp = _components_for_process_label(process_label)
        if comp is None and process_label:
            # Fail fast: require explicit mapping for new/unknown process labels
            raise ValueError(
                f"Process label '{process_label}' has no TECH_COMPONENT_MAP entry; add mapping before using it in config"
            )

        if comp:
            links.update(comp.get("links", ()))
            stores.update(comp.get("stores", ()))
            # Include canonical buses from mapping, but avoid adding energy-carrier
            # buses (e.g., renewable_electricity, grid_electricity) unless the
            # stage explicitly declares them as energy inputs. This prevents
            # slicers from preserving unused energy buses for stages that only
            # consume material inputs (e.g., steel stage using grid_electricity
            # only when declared).
            declared_energy_norm = {_normalize_commodity(e) for e in energy}
            for b in comp.get("buses", ()):
                normb = _normalize_commodity(b)
                # If this is an energy input carrier, only keep it when declared
                if normb in ENERGY_INPUTS and normb not in declared_energy_norm:
                    continue
                buses.add(normb)
    # If this stage-group uses renewable electricity, include battery
    # storage and bus as an explicit component so slicers keep batteries
    # for renewable-based stages. The user requested batteries be explicit
    # in stage configurations; adding them here maintains backward
    # compatibility while keeping per-stage skeletons functional.
    if has_renewables:
        stores.add("battery")
        buses.add("battery")
        links.update({"batt_charge", "batt_discharge"})
    return {
        "links": links,
        "stores": stores,
        "buses": buses,
        "has_renewables": has_renewables,
    }


def get_external_material_inputs(config: Dict, product: str) -> List[str]:
    """Return material buses that must be supplied externally for a stage-group.

    Inputs produced by earlier stages in the same group are not returned.
    """

    chain = get_trade_chain(config)
    groups = get_stage_groups(chain)

    target_group: Optional[Dict] = None
    for group in groups:
        if group["label"] == product:
            target_group = group
            break

    if target_group is None:
        raise ValueError(f"Product '{product}' not found in configured stage groups")

    produced = set()
    external_materials: List[str] = []

    for stage in target_group["stages"]:
        materials, _ = split_stage_inputs(stage)
        for material in materials:
            norm_material = _normalize_commodity(material)
            if (
                norm_material
                and norm_material not in produced
                and norm_material not in external_materials
            ):
                external_materials.append(norm_material)

        output_commodity = _normalize_commodity(stage.get("output_commodity", ""))
        if output_commodity:
            produced.add(output_commodity)

    return external_materials
