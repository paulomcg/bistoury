"""
Tests for Signal Manager Narrative Preservation System - Task 9.3

Tests dual-path processing to preserve TradingNarrative objects for future evolution.
"""

import asyncio
import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

from bistoury.signal_manager.narrative_buffer import (
    NarrativeBuffer, NarrativeBufferConfig, NarrativeCompressor, NarrativeIndexer,
    NarrativeContinuityTracker, NarrativeArchiver, NarrativeMetadata, NarrativeTimeline,
    NarrativeCompressionLevel, NarrativeContinuity, NarrativeChunk
)
from bistoury.strategies.narrative_generator import TradingNarrative
from bistoury.models.signals import SignalDirection


@pytest.fixture
def sample_narrative():
    """Sample TradingNarrative for testing"""
    return TradingNarrative(
        symbol="BTCUSD",
        executive_summary="BTCUSD showing strong bullish momentum",
        market_overview="Market in uptrend with volume support",
        pattern_analysis="Hammer pattern formed at support level",
        volume_analysis="Volume spike confirms pattern validity",
        risk_assessment="Low risk entry with tight stop loss",
        entry_strategy="Enter long at $50000 on breakout confirmation",
        exit_strategy="Take profit at $52000, stop loss at $49500",
        confidence_rationale="Multiple timeframes aligned bullish",
        generation_timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def config():
    """Test configuration"""
    return NarrativeBufferConfig(
        max_memory_narratives=100,
        compression_level=NarrativeCompressionLevel.LIGHT,
        background_compression=False,  # Disable for testing
        archive_after_hours=48,  # Longer for testing
        cleanup_after_days=7
    )


@pytest.fixture
def temp_storage():
    """Temporary storage directory"""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup handled by tempfile


class TestNarrativeCompressor:
    """Test narrative compression functionality"""
    
    def test_compressor_creation(self, config):
        """Test compressor creation"""
        compressor = NarrativeCompressor(config)
        assert compressor is not None
        assert compressor.config == config
    
    def test_compression_none(self, config, sample_narrative):
        """Test no compression"""
        compressor = NarrativeCompressor(config)
        compressed_data, ratio = compressor.compress_narrative(sample_narrative, NarrativeCompressionLevel.NONE)
        
        assert compressed_data is not None
        assert ratio == 1.0  # No compression
        
        # Test decompression
        decompressed = compressor.decompress_narrative(compressed_data, NarrativeCompressionLevel.NONE)
        assert decompressed.executive_summary == sample_narrative.executive_summary
    
    def test_compression_light(self, config, sample_narrative):
        """Test light compression"""
        compressor = NarrativeCompressor(config)
        compressed_data, ratio = compressor.compress_narrative(sample_narrative, NarrativeCompressionLevel.LIGHT)
        
        assert compressed_data is not None
        assert 0 < ratio <= 1.0  # Some compression
        
        # Test decompression
        decompressed = compressor.decompress_narrative(compressed_data, NarrativeCompressionLevel.LIGHT)
        assert decompressed.executive_summary == sample_narrative.executive_summary


class TestNarrativeIndexer:
    """Test narrative indexing functionality"""
    
    def test_indexer_creation(self, config):
        """Test indexer creation"""
        indexer = NarrativeIndexer(config)
        assert indexer is not None
    
    def test_add_narrative(self, config, sample_narrative):
        """Test adding narrative to index"""
        indexer = NarrativeIndexer(config)
        narrative_id = "test_narrative_1"
        
        # Create metadata for the narrative
        metadata = NarrativeMetadata(
            narrative_id=narrative_id,
            symbol="BTCUSD",
            direction=SignalDirection.BUY,
            confidence=75.0,
            strategy_id="test_strategy",
            timestamp=datetime.now(timezone.utc),
            timeframe="15m",
            keywords=["bullish", "momentum"],
            themes=["trend"]
        )
        
        indexer.add_narrative(narrative_id, metadata)
        
        # Check if narrative was indexed
        assert len(indexer.symbol_index) > 0
        assert "BTCUSD" in indexer.symbol_index
        assert narrative_id in indexer.symbol_index["BTCUSD"]


class TestNarrativeContinuityTracker:
    """Test narrative continuity tracking"""
    
    def test_tracker_creation(self, config):
        """Test tracker creation"""
        tracker = NarrativeContinuityTracker(config)
        assert tracker is not None
    
    def test_track_narrative(self, config, sample_narrative):
        """Test narrative tracking"""
        tracker = NarrativeContinuityTracker(config)
        
        # Create metadata for the narrative
        metadata = NarrativeMetadata(
            narrative_id="test_narrative_1",
            symbol="BTCUSD",
            direction=SignalDirection.BUY,
            confidence=75.0,
            strategy_id="test_strategy",
            timestamp=datetime.now(timezone.utc),
            timeframe="15m"
        )
        
        # Track the narrative
        tracker.track_narrative("test_narrative_1", metadata, sample_narrative)
        
        # Check that a story was created
        assert len(tracker.active_stories) == 1
        assert "BTCUSD" in tracker.active_stories
        
        # Get the story
        story_id = tracker.active_stories["BTCUSD"]
        story = tracker.continuity_stories[story_id]
        assert isinstance(story, NarrativeContinuity)
        assert story.symbol == "BTCUSD"
        assert len(story.narrative_sequence) == 1


@pytest.mark.asyncio
class TestNarrativeBuffer:
    """Test main narrative buffer functionality"""
    
    async def test_buffer_creation(self, config, temp_storage):
        """Test buffer creation"""
        buffer = NarrativeBuffer(config, temp_storage)
        assert buffer is not None
        
        await buffer.start()
        await buffer.stop()
    
    async def test_store_narrative(self, config, temp_storage, sample_narrative):
        """Test storing narrative"""
        buffer = NarrativeBuffer(config, temp_storage)
        await buffer.start()
        
        try:
            narrative_id = await buffer.store_narrative(
                signal_id="test_signal_1",
                narrative=sample_narrative,
                strategy_id="test_strategy",
                symbol="BTCUSD",
                direction=SignalDirection.BUY,
                confidence=75.0,
                timeframe="15m"
            )
            assert narrative_id is not None
            assert len(narrative_id) > 0
            
            # Verify narrative was stored
            retrieved = await buffer.retrieve_narrative(narrative_id)
            assert retrieved is not None
            assert retrieved.executive_summary == sample_narrative.executive_summary
        finally:
            await buffer.stop()
    
    async def test_get_timeline(self, config, temp_storage, sample_narrative):
        """Test getting narrative timeline"""
        buffer = NarrativeBuffer(config, temp_storage)
        await buffer.start()
        
        try:
            # Store multiple narratives
            narrative_ids = []
            for i in range(3):
                narrative = TradingNarrative(
                    symbol="BTCUSD",
                    executive_summary=f"Narrative {i}",
                    market_overview="Market analysis",
                    pattern_analysis="Pattern analysis",
                    volume_analysis="Volume analysis",
                    risk_assessment="Risk assessment",
                    entry_strategy="Entry strategy",
                    exit_strategy="Exit strategy",
                    confidence_rationale="Confidence rationale",
                    generation_timestamp=datetime.now(timezone.utc) + timedelta(minutes=i*5)
                )
                narrative_id = await buffer.store_narrative(
                    signal_id=f"test_signal_{i}",
                    narrative=narrative,
                    strategy_id="test_strategy",
                    symbol="BTCUSD",
                    direction=SignalDirection.BUY,
                    confidence=75.0 + i,
                    timeframe="15m"
                )
                narrative_ids.append(narrative_id)
            
            # Get timeline
            timeline = buffer.get_timeline(limit=10)
            assert isinstance(timeline, list)
            assert len(timeline) == 3
        finally:
            await buffer.stop()


def test_basic_functionality():
    """Basic test to ensure imports work"""
    assert NarrativeCompressionLevel.NONE == "none"
    assert NarrativeCompressionLevel.LIGHT == "light"
    assert NarrativeCompressionLevel.MEDIUM == "medium"
    assert NarrativeCompressionLevel.HIGH == "high"
