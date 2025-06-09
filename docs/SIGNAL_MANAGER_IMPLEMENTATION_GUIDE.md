# Signal Manager Implementation Guide: Bootstrap Strategy

## Overview

This guide outlines the incremental implementation strategy for the Bistoury Signal Manager, designed to **bootstrap funding** for advanced narrative-aware capabilities through initial mathematical signal aggregation profits.

## Strategic Architecture

### Phase 1: Mathematical Foundation (3 weeks, immediate ROI)
**Goal**: Generate trading profits to fund Phase 2-3 development

```
Strategies → Mathematical Aggregation → Trading Signals
    ↓              ↓                         ↓
TradingNarrative → Store for Future → AggregatedSignal → Position Manager
```

### Phase 2: Hybrid Approach (2 weeks evolution, funded by Phase 1)
**Goal**: Enhance with temporal narrative awareness

```
Strategies → Mathematical + Temporal Analysis → Enhanced Signals
    ↓              ↓                               ↓
TradingNarrative → LLM Narrative Analysis → Meta-Narrative → Position Manager
```

### Phase 3: Full Temporal Narrative (1 week evolution)
**Goal**: Revolutionary narrative-aware trading system

```
Strategies → Full Temporal Story Tracking → Context-Aware Decisions
    ↓              ↓                           ↓
TradingNarrative → 15min Story Evolution → LLM Decision Support → Position Manager
```

## Implementation Roadmap

### Task 9.1: Signal Aggregation Models (Week 1)

**Core Models to Implement:**

```python
@dataclass
class AggregatedSignal:
    """Mathematical aggregation of multiple strategy signals"""
    direction: SignalDirection
    confidence: float  # 0-100, weighted average
    weight: float      # Overall signal strength
    contributing_strategies: List[str]
    conflicts: List[SignalConflict]
    quality_score: float
    timestamp: datetime
    expiry: datetime

@dataclass  
class SignalWeight:
    """Dynamic weighting for strategy importance"""
    strategy_id: str
    base_weight: float        # Static strategy importance
    performance_modifier: float  # Based on recent success
    confidence_modifier: float   # Based on signal confidence
    final_weight: float      # Computed weight

@dataclass
class SignalConflict:
    """Conflict detection between contradictory signals"""
    strategy_a: str
    strategy_b: str
    conflict_type: ConflictType  # DIRECTION, TIMING, CONFIDENCE
    severity: float  # 0-1 conflict severity
    resolution: ConflictResolution

@dataclass
class TemporalSignalBuffer:
    """Stores signals and narratives for temporal analysis"""
    signals: deque[AggregatedSignal]  # Last 15 minutes
    narratives: deque[TradingNarrative]  # Full narrative history
    max_age: timedelta = timedelta(minutes=15)
    
    def add_signal(self, signal: AggregatedSignal, narrative: TradingNarrative):
        """Add signal and preserve narrative for future evolution"""
        pass
```

### Task 9.2: Mathematical Aggregation Engine (Week 1-2)

**Core Aggregation Logic:**

```python
class SignalAggregator:
    """Mathematical signal aggregation with conflict resolution"""
    
    def aggregate_signals(self, 
                         signals: List[GeneratedSignal]) -> AggregatedSignal:
        """
        Aggregate multiple strategy signals using weighted mathematics
        
        Algorithm:
        1. Weight signals by strategy performance and confidence
        2. Detect and resolve conflicts 
        3. Calculate aggregate confidence
        4. Generate quality score
        """
        
        # Weight calculation
        weighted_signals = []
        for signal in signals:
            weight = self._calculate_weight(signal)
            weighted_signals.append((signal, weight))
        
        # Conflict detection
        conflicts = self._detect_conflicts(weighted_signals)
        
        # Mathematical aggregation
        direction = self._aggregate_direction(weighted_signals, conflicts)
        confidence = self._aggregate_confidence(weighted_signals, conflicts)
        
        return AggregatedSignal(
            direction=direction,
            confidence=confidence,
            conflicts=conflicts,
            quality_score=self._calculate_quality(confidence, conflicts)
        )
    
    def _calculate_weight(self, signal: GeneratedSignal) -> float:
        """Calculate signal weight based on multiple factors"""
        base_weight = STRATEGY_WEIGHTS[signal.strategy_id]
        performance_mod = self._get_performance_modifier(signal.strategy_id)
        confidence_mod = signal.confidence / 100.0
        
        return base_weight * performance_mod * confidence_mod
```

### Task 9.3: Narrative Preservation System (Week 2)

**Dual-Path Processing:**

```python
class NarrativeBuffer:
    """Preserve complete TradingNarrative objects for future evolution"""
    
    def __init__(self, max_age: timedelta = timedelta(minutes=15)):
        self.narratives: deque[TradingNarrative] = deque()
        self.max_age = max_age
        
    def store_narrative(self, narrative: TradingNarrative):
        """Store narrative with temporal indexing"""
        self.narratives.append(narrative)
        self._cleanup_expired()
        
    def get_narrative_timeline(self) -> List[TradingNarrative]:
        """Get chronological narrative evolution"""
        return list(self.narratives)
        
    def analyze_evolution(self) -> NarrativeEvolution:
        """Basic evolution analysis for Phase 2 foundation"""
        # Placeholder for Phase 2 LLM integration
        return NarrativeEvolution(
            consistency_score=self._calculate_consistency(),
            major_shifts=self._detect_narrative_shifts(),
            dominant_themes=self._extract_themes()
        )

class NarrativeArchiver:
    """Long-term narrative storage with compression"""
    
    def archive_narratives(self, narratives: List[TradingNarrative]):
        """Compress and store narratives for historical analysis"""
        compressed = self._compress_narratives(narratives)
        self._store_to_database(compressed)
```

### Task 9.4: Signal Manager Core (Week 2-3)

**Main Signal Manager Implementation:**

```python
class SignalManager:
    """
    Phase 1: Mathematical aggregation with narrative preservation
    Phase 2+: Will evolve to temporal narrative awareness
    """
    
    def __init__(self):
        self.aggregator = SignalAggregator()
        self.narrative_buffer = NarrativeBuffer()
        self.signal_buffer = TemporalSignalBuffer()
        self.message_bus = MessageBus()
        
    async def process_strategy_signal(self, 
                                    signal: GeneratedSignal,
                                    narrative: TradingNarrative):
        """
        Main signal processing pipeline
        
        Phase 1: Mathematical aggregation + narrative storage
        Phase 2: Will add LLM temporal analysis
        """
        
        # Store narrative for future evolution (critical for Phase 2)
        self.narrative_buffer.store_narrative(narrative)
        
        # Get all active signals for aggregation
        active_signals = self._get_active_signals()
        active_signals.append(signal)
        
        # Mathematical aggregation (Phase 1 approach)
        aggregated = self.aggregator.aggregate_signals(active_signals)
        
        # Store in temporal buffer
        self.signal_buffer.add_signal(aggregated, narrative)
        
        # Publish to message bus for downstream consumers
        await self._publish_signal(aggregated)
        
        return aggregated
        
    def _get_active_signals(self) -> List[GeneratedSignal]:
        """Get unexpired signals from all strategies"""
        now = datetime.now(timezone.utc)
        return [s for s in self.signal_buffer.signals 
                if s.expiry > now]
                
    async def _publish_signal(self, signal: AggregatedSignal):
        """Publish aggregated signal via message bus"""
        message = create_signal_message(
            signal=signal,
            recipient=["trader_agent", "position_manager"]
        )
        await self.message_bus.publish(message)
```

### Task 9.5: Phase 2 Evolution Framework (Week 3)

**Foundation for LLM Integration:**

```python
class TemporalAnalyzer:
    """Interface for future narrative evolution tracking"""
    
    async def analyze_narrative_evolution(self, 
                                        narratives: List[TradingNarrative]) -> TemporalAnalysis:
        """
        Phase 1: Basic placeholder
        Phase 2: LLM-powered narrative evolution analysis
        """
        # Phase 1: Simple mathematical analysis
        return TemporalAnalysis(
            consistency_score=0.8,  # Placeholder
            evolution_trend="stable",
            major_shifts=[]
        )

class MetaNarrativeGenerator:
    """Interface for future LLM meta-narrative generation"""
    
    async def generate_meta_narrative(self,
                                    signals: List[AggregatedSignal],
                                    narratives: List[TradingNarrative]) -> MetaNarrative:
        """
        Phase 1: Simple text concatenation
        Phase 2: LLM-powered story synthesis
        """
        # Phase 1: Basic implementation
        return MetaNarrative(
            summary="Multiple signals detected",
            context="Market conditions vary",
            confidence_rationale="Based on mathematical aggregation"
        )

class EvolutionFramework:
    """A/B testing and ROI measurement for evolution phases"""
    
    def __init__(self):
        self.mathematical_performance = PerformanceTracker()
        self.llm_performance = PerformanceTracker()
        
    def measure_roi(self, phase: str) -> ROIMetrics:
        """Measure ROI for funding decisions"""
        if phase == "mathematical":
            return self.mathematical_performance.get_roi()
        elif phase == "llm_enhanced":
            return self.llm_performance.get_roi()
            
    def should_evolve_to_phase2(self) -> bool:
        """Decision logic for Phase 2 evolution"""
        roi = self.measure_roi("mathematical")
        return (roi.daily_return > 0.001 and  # 0.1% daily minimum
                roi.total_profit > 5000)      # $5k funding threshold
```

## ROI Projections and Funding Strategy

### Phase 1 Revenue Model
**Conservative Estimates:**
- Trading capital: $10,000-50,000
- Expected daily return: 0.1-0.5% 
- Daily profit: $10-250
- Weekly profit: $70-1,750
- **3-week Phase 1 profit: $210-5,250**

**Break-even Analysis:**
- Development cost (3 weeks): ~$6,000 (opportunity cost)
- Break-even trading: $6,000 ÷ $100/day = 60 days
- **ROI positive if >0.2% daily returns achieved**

### Phase 2 Funding Threshold
**Evolution Trigger:**
- Minimum daily returns: 0.1% for 2 weeks
- Accumulated profit: $5,000 minimum
- System stability: 95% uptime, <5% drawdown

### Phase 3 Success Metrics
**Revenue Targets:**
- Phase 2 improvements: +50% signal quality
- Trading returns: 0.3-1% daily target
- Advanced capabilities funding: $25,000+

## Implementation Timeline

### Week 1: Foundation
- **Days 1-2**: Task 9.1 - Signal aggregation models
- **Days 3-5**: Task 9.2 - Mathematical aggregation engine
- **Weekend**: Testing and debugging

### Week 2: Core System  
- **Days 1-3**: Task 9.3 - Narrative preservation system
- **Days 4-5**: Task 9.4 - Signal Manager core implementation
- **Weekend**: Integration testing

### Week 3: Production & Evolution
- **Days 1-2**: Task 9.5 - Phase 2 evolution framework  
- **Days 3-4**: Task 9.6 - Agent integration and deployment
- **Day 5**: Production deployment and monitoring
- **Weekend**: Performance analysis and Phase 2 planning

### Week 4+: Phase 2 Decision
- **Monitor Phase 1 performance**
- **ROI assessment for Phase 2 funding**
- **Begin Task 7 (LLM Integration) if funding threshold met**

## Key Success Factors

### Technical Excellence
- **Sub-second latency** for signal aggregation
- **Robust error handling** and failover mechanisms
- **Comprehensive testing** with edge cases
- **Performance monitoring** and optimization

### Business Metrics
- **Positive ROI** within 2 weeks of deployment
- **Consistent returns** with manageable drawdown
- **Signal quality improvement** over individual strategies
- **System reliability** >95% uptime

### Evolution Readiness
- **Clean separation** between mathematical and narrative processing
- **Comprehensive narrative preservation** for Phase 2
- **Modular architecture** enabling incremental enhancement
- **A/B testing framework** for ROI measurement

## Risk Management

### Technical Risks
- **Aggregation complexity**: Start simple, evolve gradually
- **Performance bottlenecks**: Implement async processing
- **Data quality issues**: Robust validation and filtering

### Business Risks  
- **Insufficient ROI**: Conservative revenue projections
- **Market conditions**: Diversified signal sources
- **Competition**: Focus on narrative advantage

### Evolution Risks
- **Premature optimization**: Phase 1 must generate profits first
- **LLM costs**: Careful cost-benefit analysis for Phase 2
- **Complexity creep**: Maintain clean architecture boundaries

## Conclusion

This bootstrap strategy provides a **clear path to revenue** while **preserving future evolution potential**. The mathematical foundation generates immediate trading returns to fund the advanced temporal narrative capabilities that will provide long-term competitive advantage.

**Key Insight**: By preserving TradingNarrative objects in Phase 1, we maintain the rich context needed for Phase 2-3 evolution without sacrificing immediate profitability. 