"""
Signal Manager Narrative Preservation System - Task 9.3

Implements dual-path processing to preserve TradingNarrative objects for future evolution.
Core component of Phase 1 bootstrap strategy with narrative preservation for Phase 2 enhancement.
"""

import asyncio
import json
import logging
import gzip
import pickle
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Iterator
from dataclasses import dataclass, field
from enum import Enum

from pydantic import BaseModel, Field, ConfigDict
from ..strategies.narrative_generator import TradingNarrative
from ..models.signals import SignalDirection
from .models import SignalManagerConfiguration


class NarrativeCompressionLevel(str, Enum):
    """Compression levels for narrative storage"""
    NONE = "none"
    LIGHT = "light"  # Remove redundant whitespace
    MEDIUM = "medium"  # Basic text compression
    HIGH = "high"  # Aggressive compression with gzip
    ADAPTIVE = "adaptive"  # Choose based on content size


class NarrativeMetadata(BaseModel):
    """Metadata for narrative indexing and retrieval"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    narrative_id: str
    symbol: str
    direction: SignalDirection
    confidence: float
    strategy_id: str
    timestamp: datetime
    timeframe: str
    keywords: List[str] = Field(default_factory=list)
    themes: List[str] = Field(default_factory=list)
    sentiment_score: float = 0.0
    complexity_score: float = 0.0
    storage_size: int = 0
    compression_ratio: float = 1.0
    access_count: int = 0
    last_accessed: Optional[datetime] = None


class NarrativeTimeline(BaseModel):
    """Timeline entry linking signals to narratives"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    timestamp: datetime
    signal_id: str
    narrative_id: str
    strategy_id: str
    symbol: str
    direction: SignalDirection
    confidence: float
    narrative_hash: str  # For quick comparison
    sequence_number: int
    timeframe_context: Dict[str, Any] = Field(default_factory=dict)


class NarrativeChunk(BaseModel):
    """Compressed narrative storage unit"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    chunk_id: str
    narratives: List[str]  # Serialized narratives
    metadata_ids: List[str]
    compression_level: NarrativeCompressionLevel
    original_size: int
    compressed_size: int
    timestamp_range: Tuple[datetime, datetime]
    checksum: str


class NarrativeContinuity(BaseModel):
    """Tracks narrative evolution and story continuity"""
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    story_id: str
    symbol: str
    start_time: datetime
    end_time: Optional[datetime] = None
    narrative_sequence: List[str] = Field(default_factory=list)  # narrative_ids
    evolution_points: List[datetime] = Field(default_factory=list)
    consistency_score: float = 0.0
    theme_evolution: Dict[str, List[Tuple[datetime, float]]] = Field(default_factory=dict)
    sentiment_trend: List[Tuple[datetime, float]] = Field(default_factory=list)
    market_regime_changes: List[Tuple[datetime, str]] = Field(default_factory=list)


@dataclass 
class NarrativeBufferConfig:
    """Configuration for narrative buffer system"""
    max_timeline_length: int = 10000
    max_memory_narratives: int = 1000
    compression_threshold_mb: float = 10.0
    archive_after_hours: int = 24
    cleanup_after_days: int = 30
    compression_level: NarrativeCompressionLevel = NarrativeCompressionLevel.MEDIUM
    enable_continuity_tracking: bool = True
    continuity_window_hours: int = 4
    metadata_index_size: int = 5000
    background_compression: bool = True


class NarrativeCompressor:
    """Handles narrative compression and decompression"""
    
    def __init__(self, config: NarrativeBufferConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def compress_narrative(self, narrative: TradingNarrative, level: NarrativeCompressionLevel = None) -> Tuple[bytes, float]:
        """Compress a trading narrative"""
        level = level or self.config.compression_level
        
        # Serialize narrative to JSON
        narrative_dict = narrative.model_dump()
        json_data = json.dumps(narrative_dict, default=str, separators=(',', ':'))
        original_size = len(json_data.encode('utf-8'))
        
        if level == NarrativeCompressionLevel.NONE:
            compressed_data = json_data.encode('utf-8')
        elif level == NarrativeCompressionLevel.LIGHT:
            # Remove extra whitespace
            compressed_json = json.dumps(narrative_dict, default=str, separators=(',', ':'))
            compressed_data = compressed_json.encode('utf-8')
        elif level == NarrativeCompressionLevel.MEDIUM:
            # Basic compression
            compressed_data = gzip.compress(json_data.encode('utf-8'), compresslevel=6)
        elif level == NarrativeCompressionLevel.HIGH:
            # Aggressive compression
            compressed_data = gzip.compress(json_data.encode('utf-8'), compresslevel=9)
        elif level == NarrativeCompressionLevel.ADAPTIVE:
            # Choose based on size
            if original_size < 1024:  # < 1KB
                compressed_data = json_data.encode('utf-8')
            elif original_size < 10240:  # < 10KB
                compressed_data = gzip.compress(json_data.encode('utf-8'), compresslevel=6)
            else:
                compressed_data = gzip.compress(json_data.encode('utf-8'), compresslevel=9)
        else:
            compressed_data = json_data.encode('utf-8')
        
        compression_ratio = len(compressed_data) / original_size if original_size > 0 else 1.0
        return compressed_data, compression_ratio
    
    def decompress_narrative(self, compressed_data: bytes, level: NarrativeCompressionLevel) -> TradingNarrative:
        """Decompress a trading narrative"""
        try:
            if level in [NarrativeCompressionLevel.NONE, NarrativeCompressionLevel.LIGHT]:
                json_data = compressed_data.decode('utf-8')
            else:
                json_data = gzip.decompress(compressed_data).decode('utf-8')
            
            narrative_dict = json.loads(json_data)
            return TradingNarrative.model_validate(narrative_dict)
        except Exception as e:
            self.logger.error(f"Failed to decompress narrative: {e}")
            raise


class NarrativeIndexer:
    """Indexes narratives for fast retrieval and search"""
    
    def __init__(self, config: NarrativeBufferConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Indexes
        self.metadata_index: Dict[str, NarrativeMetadata] = {}
        self.symbol_index: Dict[str, Set[str]] = defaultdict(set)  # symbol -> narrative_ids
        self.strategy_index: Dict[str, Set[str]] = defaultdict(set)  # strategy -> narrative_ids
        self.timeframe_index: Dict[str, Set[str]] = defaultdict(set)  # timeframe -> narrative_ids
        self.keyword_index: Dict[str, Set[str]] = defaultdict(set)  # keyword -> narrative_ids
        self.theme_index: Dict[str, Set[str]] = defaultdict(set)  # theme -> narrative_ids
        
        # Timeline indexes
        self.time_series_index: Dict[datetime, Set[str]] = defaultdict(set)
        self.sequence_index: Dict[int, str] = {}  # sequence_number -> narrative_id
    
    def add_narrative(self, narrative_id: str, metadata: NarrativeMetadata):
        """Add narrative to indexes"""
        self.metadata_index[narrative_id] = metadata
        
        # Update indexes
        self.symbol_index[metadata.symbol].add(narrative_id)
        self.strategy_index[metadata.strategy_id].add(narrative_id)
        self.timeframe_index[metadata.timeframe].add(narrative_id)
        
        for keyword in metadata.keywords:
            self.keyword_index[keyword.lower()].add(narrative_id)
        
        for theme in metadata.themes:
            self.theme_index[theme.lower()].add(narrative_id)
        
        # Time-based indexing (round to hour for efficiency)
        hour_key = metadata.timestamp.replace(minute=0, second=0, microsecond=0)
        self.time_series_index[hour_key].add(narrative_id)
    
    def find_narratives(self, 
                       symbol: Optional[str] = None,
                       strategy_id: Optional[str] = None,
                       timeframe: Optional[str] = None,
                       keywords: Optional[List[str]] = None,
                       themes: Optional[List[str]] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       limit: int = 100) -> List[str]:
        """Find narrative IDs matching criteria"""
        candidate_sets = []
        
        # Filter by symbol
        if symbol:
            candidate_sets.append(self.symbol_index.get(symbol, set()))
        
        # Filter by strategy
        if strategy_id:
            candidate_sets.append(self.strategy_index.get(strategy_id, set()))
        
        # Filter by timeframe
        if timeframe:
            candidate_sets.append(self.timeframe_index.get(timeframe, set()))
        
        # Filter by keywords
        if keywords:
            keyword_sets = [self.keyword_index.get(kw.lower(), set()) for kw in keywords]
            if keyword_sets:
                keyword_intersection = set.intersection(*keyword_sets)
                candidate_sets.append(keyword_intersection)
        
        # Filter by themes
        if themes:
            theme_sets = [self.theme_index.get(theme.lower(), set()) for theme in themes]
            if theme_sets:
                theme_intersection = set.intersection(*theme_sets)
                candidate_sets.append(theme_intersection)
        
        # Find intersection of all filters
        if candidate_sets:
            matching_ids = set.intersection(*candidate_sets)
        else:
            matching_ids = set(self.metadata_index.keys())
        
        # Filter by time range
        if start_time or end_time:
            time_filtered = set()
            for narrative_id in matching_ids:
                metadata = self.metadata_index.get(narrative_id)
                if metadata:
                    if start_time and metadata.timestamp < start_time:
                        continue
                    if end_time and metadata.timestamp > end_time:
                        continue
                    time_filtered.add(narrative_id)
            matching_ids = time_filtered
        
        # Sort by timestamp (newest first) and apply limit
        sorted_ids = sorted(matching_ids, 
                          key=lambda nid: self.metadata_index[nid].timestamp,
                          reverse=True)
        
        return sorted_ids[:limit]
    
    def get_metadata(self, narrative_id: str) -> Optional[NarrativeMetadata]:
        """Get metadata for a narrative"""
        metadata = self.metadata_index.get(narrative_id)
        if metadata:
            # Update access tracking
            metadata.access_count += 1
            metadata.last_accessed = datetime.now(timezone.utc)
        return metadata
    
    def update_access(self, narrative_id: str):
        """Update access statistics"""
        metadata = self.metadata_index.get(narrative_id)
        if metadata:
            metadata.access_count += 1
            metadata.last_accessed = datetime.now(timezone.utc)
    
    def cleanup_old_entries(self, cutoff_time: datetime):
        """Remove old entries from indexes"""
        ids_to_remove = []
        for narrative_id, metadata in self.metadata_index.items():
            if metadata.timestamp < cutoff_time:
                ids_to_remove.append(narrative_id)
        
        for narrative_id in ids_to_remove:
            self._remove_from_indexes(narrative_id)
    
    def _remove_from_indexes(self, narrative_id: str):
        """Remove narrative from all indexes"""
        metadata = self.metadata_index.get(narrative_id)
        if not metadata:
            return
        
        # Remove from symbol index
        self.symbol_index[metadata.symbol].discard(narrative_id)
        
        # Remove from strategy index
        self.strategy_index[metadata.strategy_id].discard(narrative_id)
        
        # Remove from timeframe index
        self.timeframe_index[metadata.timeframe].discard(narrative_id)
        
        # Remove from keyword indexes
        for keyword in metadata.keywords:
            self.keyword_index[keyword.lower()].discard(narrative_id)
        
        # Remove from theme indexes
        for theme in metadata.themes:
            self.theme_index[theme.lower()].discard(narrative_id)
        
        # Remove from time series index
        hour_key = metadata.timestamp.replace(minute=0, second=0, microsecond=0)
        self.time_series_index[hour_key].discard(narrative_id)
        
        # Remove metadata
        del self.metadata_index[narrative_id]


class NarrativeContinuityTracker:
    """Tracks narrative evolution and story continuity"""
    
    def __init__(self, config: NarrativeBufferConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.continuity_stories: Dict[str, NarrativeContinuity] = {}
        self.active_stories: Dict[str, str] = {}  # symbol -> story_id
    
    def track_narrative(self, narrative_id: str, metadata: NarrativeMetadata, narrative: TradingNarrative):
        """Track narrative for continuity analysis"""
        if not self.config.enable_continuity_tracking:
            return
        
        symbol = metadata.symbol
        current_time = metadata.timestamp
        
        # Check for active story for this symbol
        if symbol in self.active_stories:
            story_id = self.active_stories[symbol]
            story = self.continuity_stories.get(story_id)
            
            if story and self._is_story_continuous(story, current_time):
                # Continue existing story
                self._add_to_story(story, narrative_id, metadata, narrative)
            else:
                # Start new story
                self._start_new_story(symbol, narrative_id, metadata, narrative)
        else:
            # Start new story
            self._start_new_story(symbol, narrative_id, metadata, narrative)
    
    def _is_story_continuous(self, story: NarrativeContinuity, current_time: datetime) -> bool:
        """Check if narrative continues the story"""
        if not story.narrative_sequence:
            return True
        
        # Check time gap from the last narrative time
        last_time = story.end_time if story.end_time else story.start_time
        time_gap = current_time - last_time
        max_gap = timedelta(hours=self.config.continuity_window_hours)
        
        return time_gap <= max_gap
    
    def _start_new_story(self, symbol: str, narrative_id: str, metadata: NarrativeMetadata, narrative: TradingNarrative):
        """Start a new narrative story"""
        story_id = f"{symbol}_{metadata.timestamp.strftime('%Y%m%d_%H%M%S')}"
        
        story = NarrativeContinuity(
            story_id=story_id,
            symbol=symbol,
            start_time=metadata.timestamp,
            narrative_sequence=[narrative_id]
        )
        
        self._analyze_narrative_content(story, narrative, metadata.timestamp)
        
        self.continuity_stories[story_id] = story
        self.active_stories[symbol] = story_id
        
        self.logger.debug(f"Started new narrative story: {story_id}")
    
    def _add_to_story(self, story: NarrativeContinuity, narrative_id: str, metadata: NarrativeMetadata, narrative: TradingNarrative):
        """Add narrative to existing story"""
        story.narrative_sequence.append(narrative_id)
        story.end_time = metadata.timestamp
        story.evolution_points.append(metadata.timestamp)
        
        self._analyze_narrative_content(story, narrative, metadata.timestamp)
        
        # Calculate consistency score
        story.consistency_score = self._calculate_consistency_score(story)
    
    def _analyze_narrative_content(self, story: NarrativeContinuity, narrative: TradingNarrative, timestamp: datetime):
        """Analyze narrative content for themes and sentiment"""
        # Extract themes from narrative content
        content_text = f"{narrative.executive_summary} {narrative.market_overview} {narrative.pattern_analysis}"
        
        # Simple theme extraction (in Phase 2, this would use LLM)
        themes = self._extract_themes(content_text)
        for theme in themes:
            if theme not in story.theme_evolution:
                story.theme_evolution[theme] = []
            story.theme_evolution[theme].append((timestamp, 1.0))  # Simple presence score
        
        # Simple sentiment analysis (in Phase 2, this would use LLM) - use direction from metadata instead
        sentiment = self._analyze_sentiment_simple(content_text)
        story.sentiment_trend.append((timestamp, sentiment))
    
    def _extract_themes(self, content: str) -> List[str]:
        """Extract themes from narrative content (simplified for Phase 1)"""
        # Simple keyword-based theme extraction
        themes = []
        content_lower = content.lower()
        
        theme_keywords = {
            "reversal": ["reversal", "turn", "change", "shift"],
            "breakout": ["breakout", "break", "breakthrough", "resistance"],
            "trend": ["trend", "trending", "direction", "momentum"],
            "volume": ["volume", "participation", "interest"],
            "volatility": ["volatility", "volatile", "choppy", "uncertain"]
        }
        
        for theme, keywords in theme_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                themes.append(theme)
        
        return themes
    
    def _analyze_sentiment_simple(self, content: str) -> float:
        """Analyze sentiment from narrative content (simplified for Phase 1)"""
        # Simple sentiment based on confidence words
        positive_words = ["strong", "clear", "confirmed", "bullish", "positive", "excellent", "favorable"]
        negative_words = ["weak", "uncertain", "bearish", "negative", "declining", "poor", "unfavorable"]
        
        content_lower = content.lower()
        positive_score = sum(1 for word in positive_words if word in content_lower)
        negative_score = sum(1 for word in negative_words if word in content_lower)
        
        # Neutral baseline with sentiment adjustment
        base_sentiment = 0.0
        sentiment_modifier = (positive_score - negative_score) * 0.15
        
        return max(-1.0, min(1.0, base_sentiment + sentiment_modifier))
    
    def _calculate_consistency_score(self, story: NarrativeContinuity) -> float:
        """Calculate narrative consistency score"""
        if len(story.narrative_sequence) < 2:
            return 1.0
        
        # Simple consistency based on sentiment trend stability
        if len(story.sentiment_trend) < 2:
            return 1.0
        
        sentiment_values = [s[1] for s in story.sentiment_trend]
        sentiment_variance = sum((s - sum(sentiment_values) / len(sentiment_values)) ** 2 for s in sentiment_values) / len(sentiment_values)
        
        # Lower variance = higher consistency
        consistency = max(0.0, 1.0 - sentiment_variance)
        return consistency
    
    def get_story_timeline(self, symbol: str, hours_back: int = 24) -> Optional[NarrativeContinuity]:
        """Get recent story timeline for symbol"""
        story_id = self.active_stories.get(symbol)
        if not story_id:
            return None
        
        story = self.continuity_stories.get(story_id)
        if not story:
            return None
        
        # Filter to recent timeline
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        if story.start_time < cutoff_time:
            return story
        
        return story


class NarrativeArchiver:
    """Handles long-term narrative storage and archival"""
    
    def __init__(self, config: NarrativeBufferConfig, storage_path: Path):
        self.config = config
        self.storage_path = storage_path
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)
        self.compressor = NarrativeCompressor(config)
        
        # Archive indexes
        self.archived_chunks: Dict[str, NarrativeChunk] = {}
        self.chunk_index: Dict[str, str] = {}  # narrative_id -> chunk_id
    
    def archive_narratives(self, narratives: Dict[str, TradingNarrative], metadata_map: Dict[str, NarrativeMetadata]) -> str:
        """Archive a batch of narratives"""
        if not narratives:
            return ""
        
        chunk_id = f"chunk_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        # Compress narratives
        compressed_narratives = []
        total_original_size = 0
        total_compressed_size = 0
        
        for narrative_id, narrative in narratives.items():
            compressed_data, compression_ratio = self.compressor.compress_narrative(narrative)
            compressed_narratives.append(compressed_data.hex())  # Store as hex string
            
            metadata = metadata_map.get(narrative_id)
            if metadata:
                total_original_size += metadata.storage_size
                total_compressed_size += len(compressed_data)
        
        # Create chunk
        timestamps = [metadata.timestamp for metadata in metadata_map.values()]
        chunk = NarrativeChunk(
            chunk_id=chunk_id,
            narratives=compressed_narratives,
            metadata_ids=list(narratives.keys()),
            compression_level=self.config.compression_level,
            original_size=total_original_size,
            compressed_size=total_compressed_size,
            timestamp_range=(min(timestamps), max(timestamps)),
            checksum=self._calculate_checksum(compressed_narratives)
        )
        
        # Save to file
        chunk_file = self.storage_path / f"{chunk_id}.json"
        with open(chunk_file, 'w') as f:
            json.dump(chunk.model_dump(mode='json'), f, separators=(',', ':'))
        
        # Update indexes
        self.archived_chunks[chunk_id] = chunk
        for narrative_id in narratives.keys():
            self.chunk_index[narrative_id] = chunk_id
        
        self.logger.info(f"Archived {len(narratives)} narratives to {chunk_id}, compression: {total_compressed_size/total_original_size:.2%}")
        return chunk_id
    
    def retrieve_narrative(self, narrative_id: str) -> Optional[TradingNarrative]:
        """Retrieve a narrative from archive"""
        chunk_id = self.chunk_index.get(narrative_id)
        if not chunk_id:
            return None
        
        chunk = self.archived_chunks.get(chunk_id)
        if not chunk:
            # Load chunk from file
            chunk = self._load_chunk(chunk_id)
            if not chunk:
                return None
        
        # Find narrative in chunk
        try:
            narrative_index = chunk.metadata_ids.index(narrative_id)
            compressed_hex = chunk.narratives[narrative_index]
            compressed_data = bytes.fromhex(compressed_hex)
            
            return self.compressor.decompress_narrative(compressed_data, chunk.compression_level)
        except (ValueError, IndexError, Exception) as e:
            self.logger.error(f"Failed to retrieve narrative {narrative_id}: {e}")
            return None
    
    def _load_chunk(self, chunk_id: str) -> Optional[NarrativeChunk]:
        """Load chunk from file"""
        chunk_file = self.storage_path / f"{chunk_id}.json"
        if not chunk_file.exists():
            return None
        
        try:
            with open(chunk_file, 'r') as f:
                chunk_data = json.load(f)
            
            chunk = NarrativeChunk.model_validate(chunk_data)
            self.archived_chunks[chunk_id] = chunk
            return chunk
        except Exception as e:
            self.logger.error(f"Failed to load chunk {chunk_id}: {e}")
            return None
    
    def _calculate_checksum(self, compressed_narratives: List[str]) -> str:
        """Calculate checksum for data integrity"""
        import hashlib
        content = "".join(compressed_narratives)
        return hashlib.md5(content.encode()).hexdigest()
    
    def cleanup_old_archives(self, cutoff_time: datetime):
        """Clean up old archived chunks"""
        chunks_to_remove = []
        
        for chunk_id, chunk in self.archived_chunks.items():
            if chunk.timestamp_range[1] < cutoff_time:
                chunks_to_remove.append(chunk_id)
        
        for chunk_id in chunks_to_remove:
            # Remove chunk file
            chunk_file = self.storage_path / f"{chunk_id}.json"
            if chunk_file.exists():
                chunk_file.unlink()
            
            # Update indexes
            chunk = self.archived_chunks[chunk_id]
            for narrative_id in chunk.metadata_ids:
                self.chunk_index.pop(narrative_id, None)
            
            del self.archived_chunks[chunk_id]
            
            self.logger.info(f"Cleaned up archived chunk: {chunk_id}")


class NarrativeBuffer:
    """Main narrative buffer system for dual-path processing"""
    
    def __init__(self, config: NarrativeBufferConfig, storage_path: Optional[Path] = None):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Core components
        self.compressor = NarrativeCompressor(config)
        self.indexer = NarrativeIndexer(config)
        self.continuity_tracker = NarrativeContinuityTracker(config)
        
        # Storage
        storage_path = storage_path or Path("data/narratives")
        self.archiver = NarrativeArchiver(config, storage_path)
        
        # In-memory storage
        self.active_narratives: Dict[str, TradingNarrative] = {}
        self.narrative_timeline: deque[NarrativeTimeline] = deque(maxlen=config.max_timeline_length)
        
        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._compression_queue: asyncio.Queue = asyncio.Queue()
        self._sequence_counter = 0
        
        # Statistics
        self.stats = {
            "narratives_stored": 0,
            "narratives_retrieved": 0,
            "narratives_archived": 0,
            "compression_ratio": 0.0,
            "avg_retrieval_time_ms": 0.0
        }
    
    async def start(self):
        """Start background processing"""
        if self.config.background_compression:
            task = asyncio.create_task(self._background_compression_worker())
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        
        # Start cleanup task
        cleanup_task = asyncio.create_task(self._periodic_cleanup())
        self._background_tasks.add(cleanup_task)
        cleanup_task.add_done_callback(self._background_tasks.discard)
        
        self.logger.info("Narrative buffer started with background processing")
    
    async def stop(self):
        """Stop background processing"""
        for task in self._background_tasks:
            task.cancel()
        
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self.logger.info("Narrative buffer stopped")
    
    async def store_narrative(self, 
                            signal_id: str,
                            narrative: TradingNarrative,
                            strategy_id: str,
                            symbol: str,
                            direction: SignalDirection,
                            confidence: float,
                            timeframe: str = "15m") -> str:
        """Store a narrative with timeline tracking"""
        narrative_id = f"{signal_id}_{strategy_id}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')}"
        
        # Extract metadata
        metadata = self._extract_metadata(narrative_id, narrative, signal_id, strategy_id, symbol, direction, confidence, timeframe)
        
        # Store in memory
        self.active_narratives[narrative_id] = narrative
        
        # Add to indexes
        self.indexer.add_narrative(narrative_id, metadata)
        
        # Add to timeline
        timeline_entry = NarrativeTimeline(
            timestamp=metadata.timestamp,
            signal_id=signal_id,
            narrative_id=narrative_id,
            strategy_id=strategy_id,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            narrative_hash=self._calculate_narrative_hash(narrative),
            sequence_number=self._sequence_counter,
            timeframe_context={"timeframe": timeframe}
        )
        self.narrative_timeline.append(timeline_entry)
        self._sequence_counter += 1
        
        # Track continuity
        self.continuity_tracker.track_narrative(narrative_id, metadata, narrative)
        
        # Queue for compression if needed
        if self.config.background_compression:
            await self._compression_queue.put((narrative_id, metadata))
        
        self.stats["narratives_stored"] += 1
        self.logger.debug(f"Stored narrative: {narrative_id}")
        
        return narrative_id
    
    async def retrieve_narrative(self, narrative_id: str) -> Optional[TradingNarrative]:
        """Retrieve a narrative by ID"""
        start_time = datetime.now()
        
        # Check active memory first
        narrative = self.active_narratives.get(narrative_id)
        if narrative:
            self.indexer.update_access(narrative_id)
            self.stats["narratives_retrieved"] += 1
            return narrative
        
        # Check archive
        narrative = self.archiver.retrieve_narrative(narrative_id)
        if narrative:
            self.indexer.update_access(narrative_id)
            
            # Update retrieval time stats
            retrieval_time = (datetime.now() - start_time).total_seconds() * 1000
            self.stats["avg_retrieval_time_ms"] = (self.stats["avg_retrieval_time_ms"] * 0.9 + retrieval_time * 0.1)
            self.stats["narratives_retrieved"] += 1
        
        return narrative
    
    def find_narratives(self, **kwargs) -> List[str]:
        """Find narrative IDs matching criteria"""
        return self.indexer.find_narratives(**kwargs)
    
    def get_timeline(self, 
                    symbol: Optional[str] = None,
                    strategy_id: Optional[str] = None,
                    hours_back: int = 24,
                    limit: int = 100) -> List[NarrativeTimeline]:
        """Get narrative timeline"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=hours_back)
        
        timeline_entries = []
        for entry in reversed(self.narrative_timeline):
            if entry.timestamp < cutoff_time:
                break
            
            if symbol and entry.symbol != symbol:
                continue
            
            if strategy_id and entry.strategy_id != strategy_id:
                continue
            
            timeline_entries.append(entry)
            
            if len(timeline_entries) >= limit:
                break
        
        return timeline_entries
    
    def get_story_continuity(self, symbol: str, hours_back: int = 24) -> Optional[NarrativeContinuity]:
        """Get narrative story continuity for symbol"""
        return self.continuity_tracker.get_story_timeline(symbol, hours_back)
    
    def get_buffer_stats(self) -> Dict[str, Any]:
        """Get buffer statistics"""
        return {
            **self.stats,
            "active_narratives": len(self.active_narratives),
            "timeline_length": len(self.narrative_timeline),
            "indexed_narratives": len(self.indexer.metadata_index),
            "archived_chunks": len(self.archiver.archived_chunks),
            "active_stories": len(self.continuity_tracker.active_stories)
        }
    
    def _extract_metadata(self, narrative_id: str, narrative: TradingNarrative, signal_id: str,
                         strategy_id: str, symbol: str, direction: SignalDirection, 
                         confidence: float, timeframe: str) -> NarrativeMetadata:
        """Extract metadata from narrative"""
        # Simple keyword extraction
        content = f"{narrative.executive_summary} {narrative.market_overview} {narrative.pattern_analysis}"
        keywords = self._extract_keywords(content)
        themes = self._extract_themes_simple(content)
        
        # Calculate size
        narrative_json = narrative.model_dump_json()
        storage_size = len(narrative_json.encode('utf-8'))
        
        return NarrativeMetadata(
            narrative_id=narrative_id,
            symbol=symbol,
            direction=direction,
            confidence=confidence,
            strategy_id=strategy_id,
            timestamp=narrative.generation_timestamp,
            timeframe=timeframe,
            keywords=keywords,
            themes=themes,
            sentiment_score=self._calculate_sentiment_score(content, direction),
            complexity_score=self._calculate_complexity_score(narrative),
            storage_size=storage_size
        )
    
    def _extract_keywords(self, content: str) -> List[str]:
        """Extract keywords from content"""
        # Simple keyword extraction
        important_words = []
        content_lower = content.lower()
        
        trading_keywords = [
            "support", "resistance", "breakout", "reversal", "trend", "momentum",
            "volume", "pattern", "signal", "bullish", "bearish", "consolidation"
        ]
        
        for keyword in trading_keywords:
            if keyword in content_lower:
                important_words.append(keyword)
        
        return important_words[:10]  # Limit to top 10
    
    def _extract_themes_simple(self, content: str) -> List[str]:
        """Simple theme extraction"""
        themes = []
        content_lower = content.lower()
        
        if any(word in content_lower for word in ["break", "breakthrough", "resistance"]):
            themes.append("breakout")
        
        if any(word in content_lower for word in ["reverse", "reversal", "turn"]):
            themes.append("reversal")
        
        if any(word in content_lower for word in ["trend", "momentum", "direction"]):
            themes.append("trending")
        
        if any(word in content_lower for word in ["volume", "participation"]):
            themes.append("volume")
        
        return themes
    
    def _calculate_sentiment_score(self, content: str, direction: SignalDirection) -> float:
        """Calculate sentiment score"""
        # Use direction from metadata, not narrative
        base_sentiment = 0.5 if direction in [SignalDirection.BUY, SignalDirection.STRONG_BUY] else -0.5
        
        positive_words = ["strong", "clear", "confident", "bullish"]
        negative_words = ["weak", "uncertain", "bearish", "risky"]
        
        content_lower = content.lower()
        positive_count = sum(1 for word in positive_words if word in content_lower)
        negative_count = sum(1 for word in negative_words if word in content_lower)
        
        sentiment_adjustment = (positive_count - negative_count) * 0.1
        return max(-1.0, min(1.0, base_sentiment + sentiment_adjustment))
    
    def _calculate_complexity_score(self, narrative: TradingNarrative) -> float:
        """Calculate narrative complexity score"""
        # Simple complexity based on content length and detail
        timeframe_length = len(narrative.timeframe_analysis) if narrative.timeframe_analysis else 0
        total_length = len(narrative.executive_summary) + len(narrative.pattern_analysis) + timeframe_length
        factors_count = len(narrative.supporting_factors) + len(narrative.conflicting_factors)
        
        # Normalize to 0-1 scale
        length_score = min(1.0, total_length / 2000)  # Normalize by 2000 chars
        factors_score = min(1.0, factors_count / 10)   # Normalize by 10 factors
        
        return (length_score + factors_score) / 2
    
    def _calculate_narrative_hash(self, narrative: TradingNarrative) -> str:
        """Calculate hash for narrative comparison"""
        import hashlib
        content = f"{narrative.executive_summary}{narrative.confidence_rationale}"
        return hashlib.md5(content.encode()).hexdigest()[:8]
    
    async def _background_compression_worker(self):
        """Background worker for narrative compression and archival"""
        while True:
            try:
                # Wait for compression requests
                narrative_id, metadata = await asyncio.wait_for(
                    self._compression_queue.get(), timeout=60.0
                )
                
                # Check if narrative should be archived
                age = datetime.now(timezone.utc) - metadata.timestamp
                if age > timedelta(hours=self.config.archive_after_hours):
                    await self._archive_narrative(narrative_id, metadata)
                
            except asyncio.TimeoutError:
                # Periodic archival check
                await self._periodic_archival()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Background compression error: {e}")
                await asyncio.sleep(5)
    
    async def _archive_narrative(self, narrative_id: str, metadata: NarrativeMetadata):
        """Archive a single narrative"""
        narrative = self.active_narratives.get(narrative_id)
        if not narrative:
            return
        
        # Archive in batch for efficiency
        batch = {narrative_id: narrative}
        metadata_map = {narrative_id: metadata}
        
        chunk_id = self.archiver.archive_narratives(batch, metadata_map)
        if chunk_id:
            # Remove from active memory
            del self.active_narratives[narrative_id]
            self.stats["narratives_archived"] += 1
    
    async def _periodic_archival(self):
        """Periodic archival of old narratives"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=self.config.archive_after_hours)
        
        # Find narratives to archive
        narratives_to_archive = {}
        metadata_to_archive = {}
        
        for narrative_id in list(self.active_narratives.keys()):
            metadata = self.indexer.get_metadata(narrative_id)
            if metadata and metadata.timestamp < cutoff_time:
                narrative = self.active_narratives.get(narrative_id)
                if narrative:
                    narratives_to_archive[narrative_id] = narrative
                    metadata_to_archive[narrative_id] = metadata
        
        if narratives_to_archive:
            chunk_id = self.archiver.archive_narratives(narratives_to_archive, metadata_to_archive)
            if chunk_id:
                # Remove from active memory
                for narrative_id in narratives_to_archive:
                    del self.active_narratives[narrative_id]
                
                self.stats["narratives_archived"] += len(narratives_to_archive)
                self.logger.info(f"Archived {len(narratives_to_archive)} narratives to {chunk_id}")
    
    async def _periodic_cleanup(self):
        """Periodic cleanup of old data"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Cleanup old indexes
                cutoff_time = datetime.now(timezone.utc) - timedelta(days=self.config.cleanup_after_days)
                self.indexer.cleanup_old_entries(cutoff_time)
                
                # Cleanup old archives
                self.archiver.cleanup_old_archives(cutoff_time)
                
                self.logger.debug("Completed periodic cleanup")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}") 