"""
Test suite for Narrative Preservation System - Task 9.3

Tests narrative storage, retrieval, compression, indexing, and continuity tracking.
"""

import pytest
import asyncio
import tempfile
import json
import gzip
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import Mock, patch

from src.bistoury.signal_manager.narrative_buffer import (
    NarrativeBuffer,
    NarrativeBufferConfig,
    NarrativeCompressor,
    NarrativeIndexer,
    NarrativeContinuityTracker,
    NarrativeArchiver,
    NarrativeMetadata,
    NarrativeTimeline,
    NarrativeChunk,
    NarrativeContinuity,
    NarrativeCompressionLevel
)
from src.bistoury.strategies.narrative_generator import TradingNarrative, NarrativeStyle, NarrativeConfiguration
from src.bistoury.models.signals import SignalDirection


@pytest.fixture
def sample_narrative():
    """Create a sample trading narrative for testing"""
    return TradingNarrative(
        executive_summary="Bullish hammer pattern detected with strong volume confirmation",
        market_overview="BTC showing bullish momentum with breakout above resistance",
        pattern_analysis="Clear hammer pattern with long lower shadow indicating rejection of lower prices",
        timeframe_analysis="15m shows bullish structure, 5m confirms entry timing",
        volume_analysis="Volume spike on hammer formation confirms institutional interest",
        risk_assessment="Risk/reward ratio 1:2.5 with stop below pattern low",
        entry_strategy="Enter above hammer high at $50,150 with limit order",
        exit_strategy="Stop loss at $49,800, take profit at $51,000",
        confidence_rationale="85% confidence based on strong technical pattern, volume confirmation, and favorable market context",
        supporting_factors=["Volume confirmation", "Trend alignment", "Technical pattern"],
        conflicting_factors=["Minor resistance ahead"],
        key_warnings=["Watch for volume decline"],
        generation_timestamp=datetime.now(timezone.utc)
    )


@pytest.fixture
def buffer_config():
    """Create test configuration for narrative buffer"""
    return NarrativeBufferConfig(
        max_timeline_length=100,
        max_memory_narratives=50,
        compression_threshold_mb=1.0,
        archive_after_hours=1,
        cleanup_after_days=1,
        compression_level=NarrativeCompressionLevel.MEDIUM,
        enable_continuity_tracking=True,
        continuity_window_hours=2,
        background_compression=False  # Disable for testing
    )


class TestNarrativeCompressor:
    """Test narrative compression and decompression"""
    
    def test_compression_levels(self, buffer_config, sample_narrative):
        """Test different compression levels"""
        compressor = NarrativeCompressor(buffer_config)
        
        # Test all compression levels
        levels = [
            NarrativeCompressionLevel.NONE,
            NarrativeCompressionLevel.LIGHT,
            NarrativeCompressionLevel.MEDIUM,
            NarrativeCompressionLevel.HIGH,
            NarrativeCompressionLevel.ADAPTIVE
        ]
        
        for level in levels:
            compressed_data, compression_ratio = compressor.compress_narrative(sample_narrative, level)
            assert isinstance(compressed_data, bytes)
            assert 0.0 < compression_ratio <= 1.0
            
            # Test decompression
            decompressed = compressor.decompress_narrative(compressed_data, level)
            assert isinstance(decompressed, TradingNarrative)
            assert decompressed.symbol == sample_narrative.symbol
            assert decompressed.direction == sample_narrative.direction
            assert decompressed.confidence == sample_narrative.confidence
    
    def test_compression_ratio_improvement(self, buffer_config, sample_narrative):
        """Test that higher compression levels reduce size"""
        compressor = NarrativeCompressor(buffer_config)
        
        # Compare compression ratios
        none_data, none_ratio = compressor.compress_narrative(sample_narrative, NarrativeCompressionLevel.NONE)
        medium_data, medium_ratio = compressor.compress_narrative(sample_narrative, NarrativeCompressionLevel.MEDIUM)
        high_data, high_ratio = compressor.compress_narrative(sample_narrative, NarrativeCompressionLevel.HIGH)
        
        # Higher compression should result in smaller ratios
        assert medium_ratio <= none_ratio
        assert high_ratio <= medium_ratio
    
    def test_adaptive_compression(self, buffer_config):
        """Test adaptive compression chooses appropriate level"""
        compressor = NarrativeCompressor(buffer_config)
        
        # Small narrative
        small_narrative = TradingNarrative(
            executive_summary="Short summary",
            market_overview="Brief overview",
            pattern_analysis="Simple pattern",
            timeframe_analysis="Quick analysis",
            volume_analysis="Low volume",
            risk_assessment="Low risk",
            entry_strategy="Simple entry at market",
            exit_strategy="Quick exit on reversal",
            confidence_rationale="75% confidence from basic pattern",
            supporting_factors=["Factor1"],
            conflicting_factors=[],
            key_warnings=[],
            generation_timestamp=datetime.now(timezone.utc)
        )
        
        adaptive_data, adaptive_ratio = compressor.compress_narrative(small_narrative, NarrativeCompressionLevel.ADAPTIVE)
        assert isinstance(adaptive_data, bytes)
        assert adaptive_ratio > 0


class TestNarrativeIndexer:
    """Test narrative indexing and search functionality"""
    
    def test_indexer_creation(self, buffer_config):
        """Test indexer initialization"""
        indexer = NarrativeIndexer(buffer_config)
        
        assert isinstance(indexer.metadata_index, dict)
        assert isinstance(indexer.symbol_index, dict)
        assert isinstance(indexer.strategy_index, dict)
        assert len(indexer.metadata_index) == 0
    
    def test_add_narrative_to_index(self, buffer_config):
        """Test adding narrative metadata to indexes"""
        indexer = NarrativeIndexer(buffer_config)
        
        metadata = NarrativeMetadata(
            narrative_id="test_1",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=85.0,
            strategy_id="candlestick",
            timestamp=datetime.now(timezone.utc),
            timeframe="15m",
            keywords=["hammer", "bullish", "volume"],
            themes=["reversal", "breakout"]
        )
        
        indexer.add_narrative("test_1", metadata)
        
        # Check indexes
        assert "test_1" in indexer.metadata_index
        assert "test_1" in indexer.symbol_index["BTC"]
        assert "test_1" in indexer.strategy_index["candlestick"]
        assert "test_1" in indexer.timeframe_index["15m"]
        assert "test_1" in indexer.keyword_index["hammer"]
        assert "test_1" in indexer.theme_index["reversal"]
    
    def test_find_narratives_by_symbol(self, buffer_config):
        """Test finding narratives by symbol"""
        indexer = NarrativeIndexer(buffer_config)
        
        # Add multiple narratives
        for i, symbol in enumerate(["BTC", "ETH", "BTC"]):
            metadata = NarrativeMetadata(
                narrative_id=f"test_{i}",
                symbol=symbol,
                direction=SignalDirection.BUY,
                confidence=80.0,
                strategy_id="test",
                timestamp=datetime.now(timezone.utc),
                timeframe="15m"
            )
            indexer.add_narrative(f"test_{i}", metadata)
        
        # Find BTC narratives
        btc_narratives = indexer.find_narratives(symbol="BTC")
        assert len(btc_narratives) == 2
        assert "test_0" in btc_narratives
        assert "test_2" in btc_narratives
        
        # Find ETH narratives
        eth_narratives = indexer.find_narratives(symbol="ETH")
        assert len(eth_narratives) == 1
        assert "test_1" in eth_narratives
    
    def test_find_narratives_by_time_range(self, buffer_config):
        """Test finding narratives by time range"""
        indexer = NarrativeIndexer(buffer_config)
        
        base_time = datetime.now(timezone.utc)
        
        # Add narratives with different timestamps
        for i in range(3):
            metadata = NarrativeMetadata(
                narrative_id=f"test_{i}",
                symbol="BTC",
                direction=SignalDirection.BUY,
                confidence=80.0,
                strategy_id="test",
                timestamp=base_time + timedelta(hours=i),
                timeframe="15m"
            )
            indexer.add_narrative(f"test_{i}", metadata)
        
        # Find narratives in time range
        start_time = base_time + timedelta(minutes=30)
        end_time = base_time + timedelta(hours=1, minutes=30)
        
        results = indexer.find_narratives(start_time=start_time, end_time=end_time)
        assert len(results) == 1
        assert "test_1" in results
    
    def test_find_narratives_by_keywords(self, buffer_config):
        """Test finding narratives by keywords"""
        indexer = NarrativeIndexer(buffer_config)
        
        # Add narratives with different keywords
        metadata1 = NarrativeMetadata(
            narrative_id="test_1",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=80.0,
            strategy_id="test",
            timestamp=datetime.now(timezone.utc),
            timeframe="15m",
            keywords=["hammer", "bullish"]
        )
        
        metadata2 = NarrativeMetadata(
            narrative_id="test_2",
            symbol="BTC",
            direction=SignalDirection.SELL,
            confidence=75.0,
            strategy_id="test",
            timestamp=datetime.now(timezone.utc),
            timeframe="15m",
            keywords=["shooting_star", "bearish"]
        )
        
        indexer.add_narrative("test_1", metadata1)
        indexer.add_narrative("test_2", metadata2)
        
        # Find by keyword
        hammer_results = indexer.find_narratives(keywords=["hammer"])
        assert len(hammer_results) == 1
        assert "test_1" in hammer_results
        
        bearish_results = indexer.find_narratives(keywords=["bearish"])
        assert len(bearish_results) == 1
        assert "test_2" in bearish_results
    
    def test_access_tracking(self, buffer_config):
        """Test access count tracking"""
        indexer = NarrativeIndexer(buffer_config)
        
        metadata = NarrativeMetadata(
            narrative_id="test_1",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=80.0,
            strategy_id="test",
            timestamp=datetime.now(timezone.utc),
            timeframe="15m"
        )
        
        indexer.add_narrative("test_1", metadata)
        
        # Check initial access count
        retrieved_metadata = indexer.get_metadata("test_1")
        assert retrieved_metadata.access_count == 1
        
        # Update access and check again
        indexer.update_access("test_1")
        updated_metadata = indexer.get_metadata("test_1")
        assert updated_metadata.access_count == 3  # 1 initial + 1 from get_metadata + 1 from update_access


class TestNarrativeContinuityTracker:
    """Test narrative continuity and story evolution tracking"""
    
    def test_continuity_tracker_creation(self, buffer_config):
        """Test continuity tracker initialization"""
        tracker = NarrativeContinuityTracker(buffer_config)
        
        assert isinstance(tracker.continuity_stories, dict)
        assert isinstance(tracker.active_stories, dict)
        assert len(tracker.continuity_stories) == 0
    
    def test_start_new_story(self, buffer_config, sample_narrative):
        """Test starting a new narrative story"""
        tracker = NarrativeContinuityTracker(buffer_config)
        
        metadata = NarrativeMetadata(
            narrative_id="test_1",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=85.0,
            strategy_id="candlestick",
            timestamp=datetime.now(timezone.utc),
            timeframe="15m"
        )
        
        tracker.track_narrative("test_1", metadata, sample_narrative)
        
        # Check story was created
        assert "BTC" in tracker.active_stories
        story_id = tracker.active_stories["BTC"]
        assert story_id in tracker.continuity_stories
        
        story = tracker.continuity_stories[story_id]
        assert story.symbol == "BTC"
        assert len(story.narrative_sequence) == 1
        assert "test_1" in story.narrative_sequence
    
    def test_continue_existing_story(self, buffer_config, sample_narrative):
        """Test continuing an existing narrative story"""
        tracker = NarrativeContinuityTracker(buffer_config)
        
        base_time = datetime.now(timezone.utc)
        
        # First narrative
        metadata1 = NarrativeMetadata(
            narrative_id="test_1",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=85.0,
            strategy_id="candlestick",
            timestamp=base_time,
            timeframe="15m"
        )
        
        tracker.track_narrative("test_1", metadata1, sample_narrative)
        
        # Second narrative (within continuity window)
        metadata2 = NarrativeMetadata(
            narrative_id="test_2",
            symbol="BTC",
            direction=SignalDirection.STRONG_BUY,
            confidence=90.0,
            strategy_id="candlestick",
            timestamp=base_time + timedelta(minutes=30),
            timeframe="15m"
        )
        
        tracker.track_narrative("test_2", metadata2, sample_narrative)
        
        # Check story continuation
        story_id = tracker.active_stories["BTC"]
        story = tracker.continuity_stories[story_id]
        
        assert len(story.narrative_sequence) == 2
        assert "test_1" in story.narrative_sequence
        assert "test_2" in story.narrative_sequence
        assert len(story.evolution_points) == 1  # One evolution point for second narrative
    
    def test_story_discontinuity(self, buffer_config, sample_narrative):
        """Test narrative story discontinuity (time gap too large)"""
        tracker = NarrativeContinuityTracker(buffer_config)
        
        base_time = datetime.now(timezone.utc)
        
        # First narrative
        metadata1 = NarrativeMetadata(
            narrative_id="test_1",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=85.0,
            strategy_id="candlestick",
            timestamp=base_time,
            timeframe="15m"
        )
        
        tracker.track_narrative("test_1", metadata1, sample_narrative)
        
        # Second narrative (outside continuity window)
        metadata2 = NarrativeMetadata(
            narrative_id="test_2",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=80.0,
            strategy_id="candlestick",
            timestamp=base_time + timedelta(hours=5),  # Beyond continuity window
            timeframe="15m"
        )
        
        tracker.track_narrative("test_2", metadata2, sample_narrative)
        
        # Should create new story
        assert len(tracker.continuity_stories) == 2
        
        # Check that we have a new active story
        story_id = tracker.active_stories["BTC"]
        story = tracker.continuity_stories[story_id]
        assert len(story.narrative_sequence) == 1
        assert "test_2" in story.narrative_sequence
    
    def test_get_story_timeline(self, buffer_config, sample_narrative):
        """Test retrieving story timeline"""
        tracker = NarrativeContinuityTracker(buffer_config)
        
        metadata = NarrativeMetadata(
            narrative_id="test_1",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=85.0,
            strategy_id="candlestick",
            timestamp=datetime.now(timezone.utc),
            timeframe="15m"
        )
        
        tracker.track_narrative("test_1", metadata, sample_narrative)
        
        # Get timeline
        timeline = tracker.get_story_timeline("BTC")
        assert timeline is not None
        assert timeline.symbol == "BTC"
        assert len(timeline.narrative_sequence) == 1


class TestNarrativeArchiver:
    """Test narrative archival and retrieval"""
    
    def test_archiver_creation(self, buffer_config):
        """Test archiver initialization"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            archiver = NarrativeArchiver(buffer_config, storage_path)
            
            assert archiver.storage_path == storage_path
            assert isinstance(archiver.archived_chunks, dict)
            assert isinstance(archiver.chunk_index, dict)
    
    def test_archive_narratives(self, buffer_config, sample_narrative):
        """Test archiving narratives"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            archiver = NarrativeArchiver(buffer_config, storage_path)
            
            # Create test data
            narratives = {"test_1": sample_narrative}
            metadata_map = {
                "test_1": NarrativeMetadata(
                    narrative_id="test_1",
                    symbol="BTC",
                    direction=SignalDirection.BUY,
                    confidence=85.0,
                    strategy_id="candlestick",
                    timestamp=datetime.now(timezone.utc),
                    timeframe="15m",
                    storage_size=1000
                )
            }
            
            # Archive narratives
            chunk_id = archiver.archive_narratives(narratives, metadata_map)
            
            assert chunk_id != ""
            assert chunk_id in archiver.archived_chunks
            assert "test_1" in archiver.chunk_index
            
            # Check chunk file was created
            chunk_file = storage_path / f"{chunk_id}.json"
            assert chunk_file.exists()
    
    def test_retrieve_archived_narrative(self, buffer_config, sample_narrative):
        """Test retrieving narratives from archive"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            archiver = NarrativeArchiver(buffer_config, storage_path)
            
            # Archive narrative
            narratives = {"test_1": sample_narrative}
            metadata_map = {
                "test_1": NarrativeMetadata(
                    narrative_id="test_1",
                    symbol="BTC",
                    direction=SignalDirection.BUY,
                    confidence=85.0,
                    strategy_id="candlestick",
                    timestamp=datetime.now(timezone.utc),
                    timeframe="15m",
                    storage_size=1000
                )
            }
            
            chunk_id = archiver.archive_narratives(narratives, metadata_map)
            
            # Retrieve narrative
            retrieved = archiver.retrieve_narrative("test_1")
            
            assert retrieved is not None
            assert retrieved.symbol == sample_narrative.symbol
            assert retrieved.direction == sample_narrative.direction
            assert retrieved.confidence == sample_narrative.confidence


class TestNarrativeBuffer:
    """Test main narrative buffer functionality"""
    
    @pytest.fixture
    def narrative_buffer(self, buffer_config):
        """Create narrative buffer for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            buffer = NarrativeBuffer(buffer_config, storage_path)
            yield buffer
    
    @pytest.mark.asyncio
    async def test_buffer_lifecycle(self, narrative_buffer):
        """Test buffer start and stop"""
        await narrative_buffer.start()
        
        # Check background tasks are running
        assert len(narrative_buffer._background_tasks) > 0
        
        await narrative_buffer.stop()
        
        # Check background tasks are stopped
        assert len(narrative_buffer._background_tasks) == 0
    
    @pytest.mark.asyncio
    async def test_store_narrative(self, narrative_buffer, sample_narrative):
        """Test storing a narrative"""
        narrative_id = await narrative_buffer.store_narrative(
            signal_id="signal_1",
            narrative=sample_narrative,
            strategy_id="candlestick",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=85.0,
            timeframe="15m"
        )
        
        assert narrative_id != ""
        assert narrative_id in narrative_buffer.active_narratives
        assert narrative_id in narrative_buffer.indexer.metadata_index
        assert len(narrative_buffer.narrative_timeline) == 1
        
        # Check stats
        assert narrative_buffer.stats["narratives_stored"] == 1
    
    @pytest.mark.asyncio
    async def test_retrieve_narrative(self, narrative_buffer, sample_narrative):
        """Test retrieving a narrative"""
        # Store narrative first
        narrative_id = await narrative_buffer.store_narrative(
            signal_id="signal_1",
            narrative=sample_narrative,
            strategy_id="candlestick",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=85.0
        )
        
        # Retrieve narrative
        retrieved = await narrative_buffer.retrieve_narrative(narrative_id)
        
        assert retrieved is not None
        assert retrieved.symbol == sample_narrative.symbol
        assert retrieved.direction == sample_narrative.direction
        assert retrieved.confidence == sample_narrative.confidence
        
        # Check stats
        assert narrative_buffer.stats["narratives_retrieved"] == 1
    
    @pytest.mark.asyncio
    async def test_find_narratives(self, narrative_buffer, sample_narrative):
        """Test finding narratives by criteria"""
        # Store multiple narratives
        await narrative_buffer.store_narrative(
            signal_id="signal_1",
            narrative=sample_narrative,
            strategy_id="candlestick",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=85.0
        )
        
        await narrative_buffer.store_narrative(
            signal_id="signal_2",
            narrative=sample_narrative,
            strategy_id="funding_rate",
            symbol="ETH",
            direction=SignalDirection.SELL,
            confidence=75.0
        )
        
        # Find by symbol
        btc_narratives = narrative_buffer.find_narratives(symbol="BTC")
        assert len(btc_narratives) == 1
        
        eth_narratives = narrative_buffer.find_narratives(symbol="ETH")
        assert len(eth_narratives) == 1
        
        # Find by strategy
        candlestick_narratives = narrative_buffer.find_narratives(strategy_id="candlestick")
        assert len(candlestick_narratives) == 1
    
    @pytest.mark.asyncio
    async def test_get_timeline(self, narrative_buffer, sample_narrative):
        """Test getting narrative timeline"""
        # Store narratives
        await narrative_buffer.store_narrative(
            signal_id="signal_1",
            narrative=sample_narrative,
            strategy_id="candlestick",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=85.0
        )
        
        await narrative_buffer.store_narrative(
            signal_id="signal_2",
            narrative=sample_narrative,
            strategy_id="candlestick",
            symbol="BTC",
            direction=SignalDirection.STRONG_BUY,
            confidence=90.0
        )
        
        # Get timeline
        timeline = narrative_buffer.get_timeline(symbol="BTC")
        assert len(timeline) == 2
        
        # Check timeline order (newest first)
        assert timeline[0].confidence == 90.0  # Second narrative
        assert timeline[1].confidence == 85.0  # First narrative
    
    @pytest.mark.asyncio
    async def test_get_story_continuity(self, narrative_buffer, sample_narrative):
        """Test getting story continuity"""
        # Store narrative
        await narrative_buffer.store_narrative(
            signal_id="signal_1",
            narrative=sample_narrative,
            strategy_id="candlestick",
            symbol="BTC",
            direction=SignalDirection.BUY,
            confidence=85.0
        )
        
        # Get story continuity
        story = narrative_buffer.get_story_continuity("BTC")
        assert story is not None
        assert story.symbol == "BTC"
        assert len(story.narrative_sequence) == 1
    
    def test_get_buffer_stats(self, narrative_buffer):
        """Test getting buffer statistics"""
        stats = narrative_buffer.get_buffer_stats()
        
        assert "narratives_stored" in stats
        assert "narratives_retrieved" in stats
        assert "active_narratives" in stats
        assert "timeline_length" in stats
        assert "indexed_narratives" in stats
        
        assert stats["active_narratives"] == 0
        assert stats["timeline_length"] == 0


class TestIntegrationScenarios:
    """Test complete integration scenarios"""
    
    @pytest.mark.asyncio
    async def test_complete_narrative_lifecycle(self, buffer_config, sample_narrative):
        """Test complete narrative lifecycle from storage to archival"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            buffer = NarrativeBuffer(buffer_config, storage_path)
            
            await buffer.start()
            
            try:
                # Store narrative
                narrative_id = await buffer.store_narrative(
                    signal_id="signal_1",
                    narrative=sample_narrative,
                    strategy_id="candlestick",
                    symbol="BTC",
                    direction=SignalDirection.BUY,
                    confidence=85.0
                )
                
                # Retrieve immediately
                retrieved = await buffer.retrieve_narrative(narrative_id)
                assert retrieved is not None
                
                # Find narrative
                found = buffer.find_narratives(symbol="BTC")
                assert narrative_id in found
                
                # Check timeline
                timeline = buffer.get_timeline(symbol="BTC")
                assert len(timeline) == 1
                
                # Check continuity
                story = buffer.get_story_continuity("BTC")
                assert story is not None
                
                # Check stats
                stats = buffer.get_buffer_stats()
                assert stats["narratives_stored"] == 1
                assert stats["narratives_retrieved"] == 1
                assert stats["active_narratives"] == 1
                
            finally:
                await buffer.stop()
    
    @pytest.mark.asyncio
    async def test_multiple_symbol_narrative_tracking(self, buffer_config, sample_narrative):
        """Test tracking narratives across multiple symbols"""
        with tempfile.TemporaryDirectory() as temp_dir:
            storage_path = Path(temp_dir)
            buffer = NarrativeBuffer(buffer_config, storage_path)
            
            symbols = ["BTC", "ETH", "SOL"]
            
            # Store narratives for different symbols
            for i, symbol in enumerate(symbols):
                modified_narrative = sample_narrative.model_copy(update={"symbol": symbol})
                
                await buffer.store_narrative(
                    signal_id=f"signal_{i}",
                    narrative=modified_narrative,
                    strategy_id="candlestick",
                    symbol=symbol,
                    direction=SignalDirection.BUY,
                    confidence=80.0 + i * 5
                )
            
            # Check each symbol has its narrative
            for symbol in symbols:
                found = buffer.find_narratives(symbol=symbol)
                assert len(found) == 1
                
                timeline = buffer.get_timeline(symbol=symbol)
                assert len(timeline) == 1
                
                story = buffer.get_story_continuity(symbol)
                assert story is not None
                assert story.symbol == symbol
            
            # Check total stats
            stats = buffer.get_buffer_stats()
            assert stats["narratives_stored"] == 3
            assert stats["active_narratives"] == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 