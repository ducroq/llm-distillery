# Use Generic Script with Config-Driven Approach for Training Data Preparation

**Date:** 2025-11-12
**Status:** Accepted

## Context

Initially, separate training data preparation scripts existed for each filter:
- `scripts/prepare_training_data_uplifting.py` - Hardcoded for uplifting filter
- `scripts/prepare_training_data_tech_deployment.py` - Hardcoded for tech deployment

Problems with this approach:
- Code duplication (~90% identical logic)
- Maintenance burden (update multiple scripts for bug fixes)
- Scalability issue (need new script for each filter)
- Configuration duplication (dimensions, tiers hardcoded in scripts)
- Single source of truth violated (config.yaml not used)

## Decision

Create a single generic script `scripts/prepare_training_data.py` that works for any filter by reading filter configuration from `config.yaml`.

**Key principle:** `config.yaml` is the single source of truth for filter-specific information (dimensions, weights, tier boundaries, filter name).

**Script behavior:**
1. Load filter configuration from `--filter` path
2. Extract dimensions, tier boundaries, analysis field name from config
3. Apply generic logic using extracted configuration
4. Output train/val/test splits with stratification

**Usage:**
```bash
python scripts/prepare_training_data.py \
    --filter filters/{filter_name}/v1 \
    --input datasets/labeled/{filter_name}/labeled_articles.jsonl \
    --output-dir datasets/training/{filter_name}
```

## Consequences

### Positive
- **Eliminates code duplication:** Single script to maintain
- **Enforces single source of truth:** config.yaml drives behavior
- **Scalability:** New filters require NO code changes, just config
- **Consistency:** All filters prepared with identical logic
- **Easier testing:** One script to test instead of multiple
- **Better maintainability:** Bug fixes apply to all filters automatically

### Negative
- **Slight complexity increase:** Generic script more complex than hardcoded version
- **Dependency on config structure:** Requires consistent config.yaml format
- **Less obvious:** Need to read config to understand what script does

### Neutral
- Migration required: Remove old filter-specific scripts
- Documentation needed: Explain generic script approach
- Config validation: Script should validate config structure

## Alternatives Considered

- **Keep filter-specific scripts:** Rejected due to duplication and scalability issues. Would need 10+ scripts if project grows.

- **Python module with filter parameter:** Rejected as unnecessarily complex for current needs. Script approach simpler for command-line use.

- **Template-based code generation:** Rejected as overkill. Generic script simpler and more maintainable than code generation infrastructure.

## Implementation Notes

**Key functions:**
```python
def load_filter_config(filter_dir: Path) -> Dict[str, Any]:
    """Load filter configuration from config.yaml."""

def extract_filter_info(config: Dict) -> Tuple[str, List[str], Dict[str, float]]:
    """Extract filter name, dimension names, and tier boundaries."""

def get_analysis_field_name(filter_name: str) -> str:
    """Infer analysis field name from filter name.
    uplifting -> uplifting_analysis
    sustainability_tech_deployment -> sustainability_tech_deployment_analysis
    """
```

**Config.yaml requirements:**
```yaml
filter:
  name: "uplifting"  # Used to infer analysis field

scoring:
  dimensions:
    agency: {...}
    progress: {...}
    # ... (order preserved)

  tiers:
    impact:
      threshold: 7.0
    connection:
      threshold: 4.0
    # ... (sorted descending by threshold)
```

**Migration completed:**
- Created `scripts/prepare_training_data.py`
- Removed `scripts/prepare_training_data_uplifting.py`
- Removed `scripts/prepare_training_data_tech_deployment.py`
- Updated `training/README.md` with new usage

## References

- `scripts/prepare_training_data.py` - Generic script implementation
- `filters/uplifting/v1/config.yaml` - Example filter config
- `filters/sustainability_tech_deployment/v1/config.yaml` - Example filter config
- `training/README.md` - Updated usage documentation
