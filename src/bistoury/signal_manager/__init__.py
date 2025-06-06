"""
Bistoury Signal Manager Package

Multi-phase signal aggregation system:
- Phase 1: Mathematical aggregation with narrative preservation
- Phase 2: Hybrid mathematical + temporal narrative analysis  
- Phase 3: Full temporal narrative management

This package implements the bootstrap strategy for signal management,
starting with proven mathematical approaches while preserving rich
narrative context for future LLM-powered evolution.
"""

from .models import (
    AggregatedSignal,
    SignalWeight,
    SignalConflict,
    ConflictType,
    ConflictResolution,
    TemporalSignalBuffer,
    SignalQuality,
    SignalManagerConfiguration,
)

__all__ = [
    "AggregatedSignal",
    "SignalWeight", 
    "SignalConflict",
    "ConflictType",
    "ConflictResolution",
    "TemporalSignalBuffer",
    "SignalQuality",
    "SignalManagerConfiguration",
] 