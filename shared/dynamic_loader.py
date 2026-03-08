"""
================================================================================
Project VERA — Dynamic Agent Loader
================================================================================

Auto-discovers and imports agent modules from domain subfolders under
agents_logic/. Each subfolder named *_agents/ is treated as a domain.

Usage:
    from shared.dynamic_loader import discover_domain_agents

    DOMAIN_AGENTS = discover_domain_agents()
    # Returns: {
    #     "semiconductor": {
    #         "tech_spec_agent": <module>,
    #         "compliance_agent": <module>,
    #         "discrepancy_agent": <module>,
    #     },
    #     "medical": { ... },
    # }

Each module MUST have a `run(state: GraphState) -> dict` function.
================================================================================
"""

import os
import importlib


# Agent modules matching these roles are wired into the graph as nodes.
# Each domain folder should have at least these agent types:
EXPECTED_AGENT_ROLES = {
    "official_docs_agent": "retrieve_official",
    "informal_docs_agent": "retrieve_informal",
    "discrepancy_agent": "check_discrepancy",
    "db_agent": "query_database",
}


def get_available_domains(base_package: str = "agents_logic") -> list[str]:
    """
    Quickly scan for available domain folders without importing modules.

    Returns a sorted list of domain names (e.g., ["medical", "semiconductor"]).
    This is lightweight — it only reads the filesystem, no module imports.
    """
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), base_package)
    domains = []
    if os.path.exists(base_dir):
        for entry in sorted(os.listdir(base_dir)):
            entry_path = os.path.join(base_dir, entry)
            if os.path.isdir(entry_path) and entry.endswith("_agents"):
                domains.append(entry.replace("_agents", ""))
    return domains


# ---------------------------------------------------------------------------
# Domain Config Loader — replaces hardcoded DOMAIN_KEYWORDS in config.py
# ---------------------------------------------------------------------------

_domain_configs_cache: dict | None = None


def load_domain_configs(base_package: str = "agents_logic") -> dict[str, dict]:
    """
    Import ``domain_config.py`` from every discovered ``{domain}_agents/``
    folder and return a merged mapping of ``{domain_name: DOMAIN_CONFIG}``.

    Each ``domain_config.py`` MUST export a ``DOMAIN_CONFIG`` dict with at
    least ``keywords`` and ``aliases`` keys.  If a domain folder has no
    ``domain_config.py``, it is silently skipped.

    Results are cached after the first call.
    """
    global _domain_configs_cache
    if _domain_configs_cache is not None:
        return _domain_configs_cache

    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), base_package)
    configs: dict[str, dict] = {}

    if not os.path.exists(base_dir):
        _domain_configs_cache = configs
        return configs

    for entry in sorted(os.listdir(base_dir)):
        entry_path = os.path.join(base_dir, entry)
        if not os.path.isdir(entry_path) or not entry.endswith("_agents"):
            continue

        domain_name = entry.replace("_agents", "")
        config_module_path = f"{base_package}.{entry}.domain_config"

        try:
            mod = importlib.import_module(config_module_path)
            cfg = getattr(mod, "DOMAIN_CONFIG", None)
            if cfg and isinstance(cfg, dict):
                configs[domain_name] = cfg
                print(f"[LOADER] 📋 Loaded domain config for '{domain_name}'")
            else:
                print(f"[LOADER] ⚠️  {config_module_path} has no valid DOMAIN_CONFIG dict")
        except ModuleNotFoundError:
            # No domain_config.py — domain is still usable, just no routing hints
            print(f"[LOADER] ℹ️  No domain_config.py for '{domain_name}' (optional)")
        except Exception as e:
            print(f"[LOADER] ❌ Failed to load config for '{domain_name}': {e}")

    _domain_configs_cache = configs
    return configs


def build_routing_heuristics(
    domain_configs: dict[str, dict] | None = None,
) -> dict[str, list[str]]:
    """
    Merge all domains' keyword dicts into a single flat lookup keyed by
    intent label.  Returns ``{"technical": [...], "compliance": [...]}``.

    Used by the Router Agent for zero-assumption keyword scoring.
    """
    if domain_configs is None:
        domain_configs = load_domain_configs()

    merged: dict[str, list[str]] = {}
    for _domain, cfg in domain_configs.items():
        for intent, keywords in cfg.get("keywords", {}).items():
            merged.setdefault(intent, []).extend(keywords)

    # Deduplicate while preserving order
    for intent in merged:
        seen: set[str] = set()
        deduped: list[str] = []
        for kw in merged[intent]:
            if kw not in seen:
                seen.add(kw)
                deduped.append(kw)
        merged[intent] = deduped

    return merged


def resolve_domain_alias(alias: str, domain_configs: dict[str, dict] | None = None) -> str | None:
    """
    Resolve a domain alias (e.g. ``"engineering"``) to its canonical domain
    name (e.g. ``"semiconductor"``).  Returns ``None`` if no match is found.
    """
    if domain_configs is None:
        domain_configs = load_domain_configs()

    alias_lower = alias.lower()
    for domain, cfg in domain_configs.items():
        if alias_lower == domain:
            return domain
        for a in cfg.get("aliases", []):
            if alias_lower == a.lower():
                return domain
    return None


def discover_domain_agents(
    base_package: str = "agents_logic",
) -> dict[str, dict]:
    """
    Scan agents_logic/<domain>_agents/ subfolders for agent modules.

    Convention:
      - Subfolder name: <domain>_agents/ (e.g., semiconductor_agents/)
      - Domain name is extracted by stripping the '_agents' suffix
      - Each .py file (except _ prefixed and __init__) is imported
      - Each module must have a `run(state) -> dict` function

    Args:
        base_package: Python package name for agents_logic (default: "agents_logic")

    Returns:
        dict: Nested mapping of {domain_name: {agent_name: module}}
              e.g., {"semiconductor": {"tech_spec_agent": <module>, ...}}
    """
    base_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), base_package)
    domain_agents = {}

    if not os.path.exists(base_dir):
        print(f"[LOADER] ⚠️  agents_logic/ directory not found at {base_dir}")
        return {}

    for entry in sorted(os.listdir(base_dir)):
        entry_path = os.path.join(base_dir, entry)

        # Only process directories ending with _agents
        if not os.path.isdir(entry_path):
            continue
        if not entry.endswith("_agents"):
            continue

        # Extract domain name: "semiconductor_agents" -> "semiconductor"
        domain_name = entry.replace("_agents", "")

        print(f"\n[LOADER] 📂 Discovered domain: {domain_name} ({entry}/)")

        agents = {}
        for filename in sorted(os.listdir(entry_path)):
            # Skip non-Python, private, and __init__ files
            if not filename.endswith(".py"):
                continue
            if filename.startswith("_"):
                continue

            module_name = filename[:-3]  # Strip .py
            full_module_path = f"{base_package}.{entry}.{module_name}"

            try:
                module = importlib.import_module(full_module_path)

                # Verify the module has a run() function
                if not hasattr(module, "run"):
                    print(f"[LOADER]   ⚠️  {module_name} — missing run() function, skipped")
                    continue

                if not callable(module.run):
                    print(f"[LOADER]   ⚠️  {module_name} — run is not callable, skipped")
                    continue

                agents[module_name] = module
                agent_label = getattr(module.run, "__agent_name__", module_name)
                print(f"[LOADER]   ✅ {module_name} → @vera_agent(\"{agent_label}\")")

            except Exception as e:
                print(f"[LOADER]   ❌ Failed to import {full_module_path}: {e}")

        domain_agents[domain_name] = agents
        print(f"[LOADER]   Total: {len(agents)} agents loaded for '{domain_name}'")

    print(f"\n[LOADER] Summary: {len(domain_agents)} domains discovered: {list(domain_agents.keys())}")
    return domain_agents


def get_agent_node_name(domain: str, agent_name: str) -> str:
    """
    Generate a LangGraph node name for a domain-specific agent.

    Convention: <domain>_<role>
    Example: "semiconductor_retrieve_specs", "medical_check_discrepancy"

    Args:
        domain: Domain name (e.g., "semiconductor")
        agent_name: Agent module name (e.g., "tech_spec_agent")

    Returns:
        str: Node name for LangGraph (e.g., "semiconductor_retrieve_specs")
    """
    role = EXPECTED_AGENT_ROLES.get(agent_name, agent_name)
    return f"{domain}_{role}"


def register_domain_nodes(workflow, domain_agents: dict) -> dict[str, list[str]]:
    """
    Register all discovered domain agents as LangGraph nodes.

    For each domain and each agent, adds a node named <domain>_<role>.

    Args:
        workflow: LangGraph StateGraph instance
        domain_agents: Output from discover_domain_agents()

    Returns:
        dict: Mapping of {domain: [node_name, ...]} for graph wiring
    """
    domain_nodes = {}

    for domain, agents in domain_agents.items():
        nodes = []
        for agent_name, module in agents.items():
            node_name = get_agent_node_name(domain, agent_name)
            workflow.add_node(node_name, module.run)
            nodes.append(node_name)
            print(f"[LOADER] Registered node: {node_name}")

        domain_nodes[domain] = nodes

    return domain_nodes
