"""
Base PreFilter Class - Backward Compatibility Module

DEPRECATED: Import from filters.common.base_prefilter instead.

This module re-exports BasePreFilter from the new location for backward compatibility.
"""

# Re-export from new location
from filters.common.base_prefilter import BasePreFilter

__all__ = ['BasePreFilter']
