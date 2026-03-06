"""Ground truth dataset generation using LLM oracles."""


def analysis_field_name(filter_name: str) -> str:
    """Standard key for oracle analysis in scored JSONL records.

    Convention: each filter's scores are stored under '{filter_name}_analysis'
    to allow multi-filter scoring on the same article. Used by batch_scorer.py
    (write) and prepare_data.py (read).

    Examples:
        >>> analysis_field_name("uplifting")
        'uplifting_analysis'
        >>> analysis_field_name("nature_recovery")
        'nature_recovery_analysis'
    """
    return f"{filter_name}_analysis"


__all__ = ["analysis_field_name"]
