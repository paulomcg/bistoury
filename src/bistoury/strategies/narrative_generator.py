"""
Narrative Generation for LLM Integration

This module generates human-readable narratives describing candlestick pattern analysis,
multi-timeframe confluence, and trading signals for consumption by Large Language Models.
The narratives provide context-rich explanations that help LLMs make informed trading decisions.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import List, Optional, Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass

from pydantic import BaseModel, Field, computed_field, ConfigDict

from ..models.signals import CandlestickPattern, CandlestickData, SignalDirection, PatternType, Timeframe
from .candlestick_models import PatternStrength, VolumeProfile, PatternQuality, MultiTimeframePattern
from .pattern_scoring import CompositePatternScore, MarketSession, VolatilityRegime, TrendStrength
from .timeframe_analyzer import TimeframeAnalysisResult
from .signal_generator import (
    GeneratedSignal, SignalTiming, RiskLevel, SignalRiskManagement, SignalEntryPoint
)


class NarrativeStyle(str, Enum):
    """Narrative style for different LLM consumption patterns."""
    TECHNICAL = "technical"      # Technical analysis focused
    CONTEXTUAL = "contextual"    # Market context focused
    COMPREHENSIVE = "comprehensive"  # Complete analysis
    CONCISE = "concise"         # Brief summary
    EDUCATIONAL = "educational"  # Explanatory style


class NarrativeSection(str, Enum):
    """Sections of market narrative."""
    MARKET_OVERVIEW = "market_overview"
    PATTERN_ANALYSIS = "pattern_analysis"
    TIMEFRAME_CONFLUENCE = "timeframe_confluence"
    VOLUME_CONFIRMATION = "volume_confirmation"
    RISK_ASSESSMENT = "risk_assessment"
    TRADING_RECOMMENDATION = "trading_recommendation"
    SUPPORTING_EVIDENCE = "supporting_evidence"
    CONFLICTING_SIGNALS = "conflicting_signals"


@dataclass
class NarrativeConfiguration:
    """Configuration for narrative generation."""
    style: NarrativeStyle = NarrativeStyle.COMPREHENSIVE
    include_technical_details: bool = True
    include_risk_metrics: bool = True
    include_confidence_scores: bool = True
    include_historical_context: bool = True
    max_pattern_count: int = 5
    focus_timeframe: Optional[Timeframe] = None
    emphasize_strengths: bool = True
    include_warnings: bool = True


class PatternNarrative(BaseModel):
    """Narrative description of a single candlestick pattern."""
    
    model_config = ConfigDict(
        json_encoders={
            Decimal: str,
            datetime: lambda dt: dt.isoformat(),
        }
    )
    
    pattern_type: PatternType = Field(..., description="Pattern type")
    timeframe: Timeframe = Field(..., description="Pattern timeframe")
    description: str = Field(..., description="Human-readable pattern description")
    significance: str = Field(..., description="Market significance explanation")
    strength_assessment: str = Field(..., description="Pattern strength evaluation")
    reliability_note: str = Field(..., description="Historical reliability information")
    context_relevance: str = Field(..., description="Current market context relevance")
    
    @classmethod
    def from_pattern(cls,
                    pattern: CandlestickPattern,
                    pattern_score: CompositePatternScore,
                    config: NarrativeConfiguration) -> "PatternNarrative":
        """Create narrative from pattern and scoring."""
        pattern_name = pattern.pattern_type.value.replace('_', ' ').title()
        timeframe_text = cls._timeframe_to_text(pattern.timeframe)
        
        # Generate description
        description = cls._generate_pattern_description(pattern, pattern_name)
        
        # Generate significance
        significance = cls._generate_significance(pattern, pattern_score)
        
        # Generate strength assessment
        strength_assessment = cls._generate_strength_assessment(pattern_score, config)
        
        # Generate reliability note
        reliability_note = cls._generate_reliability_note(pattern_score, config)
        
        # Generate context relevance
        context_relevance = cls._generate_context_relevance(pattern_score)
        
        return cls(
            pattern_type=pattern.pattern_type,
            timeframe=pattern.timeframe,
            description=f"{timeframe_text} {description}",
            significance=significance,
            strength_assessment=strength_assessment,
            reliability_note=reliability_note,
            context_relevance=context_relevance
        )
    
    @staticmethod
    def _timeframe_to_text(timeframe: Timeframe) -> str:
        """Convert timeframe to readable text."""
        mapping = {
            Timeframe.ONE_MINUTE: "1-minute",
            Timeframe.FIVE_MINUTES: "5-minute",
            Timeframe.FIFTEEN_MINUTES: "15-minute",
            Timeframe.ONE_HOUR: "1-hour",
            Timeframe.FOUR_HOURS: "4-hour",
            Timeframe.ONE_DAY: "daily"
        }
        return mapping.get(timeframe, str(timeframe))
    
    @staticmethod
    def _generate_pattern_description(pattern: CandlestickPattern, pattern_name: str) -> str:
        """Generate basic pattern description."""
        direction = "bullish" if pattern.directional_bias in [SignalDirection.BUY, SignalDirection.STRONG_BUY] else "bearish"
        
        descriptions = {
            PatternType.DOJI: f"{pattern_name} pattern indicates market indecision with opening and closing prices nearly equal",
            PatternType.HAMMER: f"{pattern_name} pattern shows potential bullish reversal with small body and long lower shadow",
            PatternType.SHOOTING_STAR: f"{pattern_name} pattern indicates potential bearish reversal with small body and long upper shadow",
            PatternType.SPINNING_TOP: f"{pattern_name} pattern suggests market uncertainty with small body and long shadows on both sides",
            PatternType.MARUBOZU: f"{pattern_name} pattern demonstrates strong {direction} momentum with little to no shadows",
            PatternType.ENGULFING: f"{pattern_name} pattern shows strong reversal as larger candle engulfs previous candle",
            PatternType.HARAMI: f"{pattern_name} pattern suggests potential reversal with inside candle following larger body",
            PatternType.PIERCING_LINE: f"{pattern_name} pattern shows bullish reversal with gap down followed by strong buying",
            PatternType.DARK_CLOUD_COVER: f"{pattern_name} pattern indicates bearish reversal with gap up followed by selling pressure",
            PatternType.MORNING_STAR: f"{pattern_name} pattern demonstrates strong bullish reversal over three candles with star pattern",
            PatternType.EVENING_STAR: f"{pattern_name} pattern shows strong bearish reversal over three candles with star formation"
        }
        
        return descriptions.get(pattern.pattern_type, f"{pattern_name} {direction} pattern detected")
    
    @staticmethod
    def _generate_significance(pattern: CandlestickPattern, pattern_score: CompositePatternScore) -> str:
        """Generate market significance explanation."""
        confidence = int(pattern_score.weighted_confidence_score)
        strength = pattern_score.pattern_strength.value.replace('_', ' ')
        
        if confidence >= 90:
            return f"Highly significant {strength} pattern with exceptional technical characteristics and strong market validation."
        elif confidence >= 80:
            return f"Very significant {strength} pattern with excellent technical quality and good market support."
        elif confidence >= 70:
            return f"Significant {strength} pattern with solid technical foundation and adequate market confirmation."
        elif confidence >= 60:
            return f"Moderately significant {strength} pattern with reasonable technical merit and some market support."
        else:
            return f"Low significance {strength} pattern with limited technical validation and weak market confirmation."
    
    @staticmethod
    def _generate_strength_assessment(pattern_score: CompositePatternScore, config: NarrativeConfiguration) -> str:
        """Generate pattern strength assessment."""
        technical = pattern_score.technical_scoring.overall_technical_score
        volume = pattern_score.volume_scoring.overall_volume_score
        context = pattern_score.context_scoring.overall_context_score
        
        assessment = f"Technical quality scores {int(technical)}/100 "
        
        if config.include_technical_details:
            if volume >= 70:
                assessment += "with strong volume confirmation, "
            elif volume >= 50:
                assessment += "with adequate volume support, "
            else:
                assessment += "with limited volume confirmation, "
            
            if context >= 70:
                assessment += "excellent market context alignment."
            elif context >= 50:
                assessment += "reasonable market context support."
            else:
                assessment += "challenging market context conditions."
        
        return assessment
    
    @staticmethod
    def _generate_reliability_note(pattern_score: CompositePatternScore, config: NarrativeConfiguration) -> str:
        """Generate reliability information."""
        historical = pattern_score.historical_performance
        success_rate = int(historical.success_rate)
        
        if not config.include_historical_context:
            return "Historical performance data available."
        
        if success_rate >= 75:
            return f"Excellent historical reliability with {success_rate}% success rate over {historical.total_occurrences} occurrences."
        elif success_rate >= 65:
            return f"Good historical performance with {success_rate}% success rate across {historical.total_occurrences} instances."
        elif success_rate >= 55:
            return f"Average historical results with {success_rate}% success rate from {historical.total_occurrences} observations."
        else:
            return f"Below-average historical performance with {success_rate}% success rate over {historical.total_occurrences} cases."
    
    @staticmethod
    def _generate_context_relevance(pattern_score: CompositePatternScore) -> str:
        """Generate market context relevance."""
        context_score = int(pattern_score.context_scoring.overall_context_score)
        
        if context_score >= 80:
            return "Highly relevant in current market conditions with excellent environmental alignment."
        elif context_score >= 60:
            return "Well-suited to current market environment with good contextual support."
        elif context_score >= 40:
            return "Moderately relevant given current market conditions with mixed environmental factors."
        else:
            return "Limited relevance in current market context with challenging environmental conditions."


class TimeframeNarrative(BaseModel):
    """Narrative for multi-timeframe analysis."""
    
    model_config = ConfigDict(
        json_encoders={
            Decimal: str,
            datetime: lambda dt: dt.isoformat(),
        }
    )
    
    confluence_summary: str = Field(..., description="Overall confluence summary")
    trend_alignment: str = Field(..., description="Trend alignment across timeframes")
    strength_distribution: str = Field(..., description="Pattern strength across timeframes")
    conflict_analysis: str = Field(..., description="Analysis of conflicting signals")
    dominant_timeframe: str = Field(..., description="Most influential timeframe")
    recommendation_basis: str = Field(..., description="Basis for trading recommendation")
    
    @classmethod
    def from_timeframe_analysis(cls,
                              analysis: TimeframeAnalysisResult,
                              config: NarrativeConfiguration) -> "TimeframeNarrative":
        """Create narrative from timeframe analysis."""
        confluence_summary = cls._generate_confluence_summary(analysis)
        trend_alignment = cls._generate_trend_alignment(analysis)
        strength_distribution = cls._generate_strength_distribution(analysis)
        conflict_analysis = cls._generate_conflict_analysis(analysis)
        dominant_timeframe = cls._identify_dominant_timeframe(analysis)
        recommendation_basis = cls._generate_recommendation_basis(analysis)
        
        return cls(
            confluence_summary=confluence_summary,
            trend_alignment=trend_alignment,
            strength_distribution=strength_distribution,
            conflict_analysis=conflict_analysis,
            dominant_timeframe=dominant_timeframe,
            recommendation_basis=recommendation_basis
        )
    
    @staticmethod
    def _generate_confluence_summary(analysis: TimeframeAnalysisResult) -> str:
        """Generate confluence summary."""
        confluence_score = int(analysis.confluence_analysis.confluence_score)
        pattern_count = len(analysis.confluence_analysis.confluence_patterns)
        
        if confluence_score >= 80:
            return f"Strong multi-timeframe confluence detected with {pattern_count} aligned patterns showing {confluence_score}% agreement."
        elif confluence_score >= 60:
            return f"Moderate confluence across timeframes with {pattern_count} patterns achieving {confluence_score}% alignment."
        elif confluence_score >= 40:
            return f"Weak confluence present with {pattern_count} patterns showing {confluence_score}% agreement across timeframes."
        else:
            return f"Limited timeframe confluence with {pattern_count} patterns achieving only {confluence_score}% alignment."
    
    @staticmethod
    def _generate_trend_alignment(analysis: TimeframeAnalysisResult) -> str:
        """Generate trend alignment narrative."""
        trend_alignment = analysis.trend_alignment
        
        if trend_alignment.alignment_score >= Decimal('0.8'):
            return f"Excellent trend alignment across all timeframes with {int(trend_alignment.alignment_score * 100)}% consistency."
        elif trend_alignment.alignment_score >= Decimal('0.6'):
            return f"Good trend alignment with {int(trend_alignment.alignment_score * 100)}% timeframe agreement on direction."
        elif trend_alignment.alignment_score >= Decimal('0.4'):
            return f"Mixed trend signals with {int(trend_alignment.alignment_score * 100)}% alignment across timeframes."
        else:
            return f"Conflicting trend directions with only {int(trend_alignment.alignment_score * 100)}% timeframe agreement."
    
    @staticmethod
    def _generate_strength_distribution(analysis: TimeframeAnalysisResult) -> str:
        """Generate strength distribution narrative."""
        patterns = analysis.confluence_analysis.confluence_patterns
        if not patterns:
            return "No significant patterns detected across timeframes."
        
        strong_patterns = sum(1 for p in patterns if p.confidence >= Decimal('80'))
        moderate_patterns = sum(1 for p in patterns if Decimal('60') <= p.confidence < Decimal('80'))
        weak_patterns = len(patterns) - strong_patterns - moderate_patterns
        
        return f"Pattern strength distribution: {strong_patterns} strong, {moderate_patterns} moderate, {weak_patterns} weak patterns across timeframes."
    
    @staticmethod
    def _generate_conflict_analysis(analysis: TimeframeAnalysisResult) -> str:
        """Generate conflict analysis."""
        if analysis.trend_alignment.conflicting_timeframes:
            conflicting = ", ".join(analysis.trend_alignment.conflicting_timeframes)
            return f"Conflicting signals detected on {conflicting} timeframes, requiring caution in position sizing."
        else:
            return "No significant conflicts detected across analyzed timeframes."
    
    @staticmethod
    def _identify_dominant_timeframe(analysis: TimeframeAnalysisResult) -> str:
        """Identify the most influential timeframe."""
        primary = analysis.confluence_analysis.primary_timeframe
        return f"{primary.value} timeframe provides the dominant signal with highest weight in analysis."
    
    @staticmethod
    def _generate_recommendation_basis(analysis: TimeframeAnalysisResult) -> str:
        """Generate basis for trading recommendation."""
        recommendation = analysis.trading_recommendation
        confidence = int(analysis.overall_confidence_score)
        
        direction_text = recommendation.value.replace('_', ' ').lower()
        return f"Trading recommendation ({direction_text}) based on {confidence}% overall confidence from multi-timeframe convergence."


class TradingNarrative(BaseModel):
    """Complete trading narrative for LLM consumption."""
    
    model_config = ConfigDict(
        json_encoders={
            Decimal: str,
            datetime: lambda dt: dt.isoformat(),
        }
    )
    
    executive_summary: str = Field(..., description="High-level trading summary")
    market_overview: str = Field(..., description="Current market condition overview")
    pattern_analysis: str = Field(..., description="Detailed pattern analysis")
    timeframe_analysis: Optional[str] = Field(None, description="Multi-timeframe analysis")
    volume_analysis: str = Field(..., description="Volume confirmation analysis")
    risk_assessment: str = Field(..., description="Risk and reward assessment")
    entry_strategy: str = Field(..., description="Entry strategy explanation")
    exit_strategy: str = Field(..., description="Exit strategy explanation")
    confidence_rationale: str = Field(..., description="Confidence level rationale")
    key_warnings: List[str] = Field(default_factory=list, description="Important warnings")
    supporting_factors: List[str] = Field(default_factory=list, description="Supporting evidence")
    conflicting_factors: List[str] = Field(default_factory=list, description="Conflicting evidence")
    
    generation_timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Narrative generation timestamp"
    )
    
    @computed_field
    @property
    def narrative_sections(self) -> Dict[str, str]:
        """Get all narrative sections as a dictionary."""
        return {
            "executive_summary": self.executive_summary,
            "market_overview": self.market_overview,
            "pattern_analysis": self.pattern_analysis,
            "timeframe_analysis": self.timeframe_analysis or "Not available",
            "volume_analysis": self.volume_analysis,
            "risk_assessment": self.risk_assessment,
            "entry_strategy": self.entry_strategy,
            "exit_strategy": self.exit_strategy,
            "confidence_rationale": self.confidence_rationale
        }
    
    @computed_field
    @property
    def full_narrative(self) -> str:
        """Generate complete narrative text."""
        sections = []
        
        sections.append(f"EXECUTIVE SUMMARY:\n{self.executive_summary}\n")
        sections.append(f"MARKET OVERVIEW:\n{self.market_overview}\n")
        sections.append(f"PATTERN ANALYSIS:\n{self.pattern_analysis}\n")
        
        if self.timeframe_analysis:
            sections.append(f"TIMEFRAME ANALYSIS:\n{self.timeframe_analysis}\n")
        
        sections.append(f"VOLUME ANALYSIS:\n{self.volume_analysis}\n")
        sections.append(f"RISK ASSESSMENT:\n{self.risk_assessment}\n")
        sections.append(f"ENTRY STRATEGY:\n{self.entry_strategy}\n")
        sections.append(f"EXIT STRATEGY:\n{self.exit_strategy}\n")
        sections.append(f"CONFIDENCE RATIONALE:\n{self.confidence_rationale}\n")
        
        if self.supporting_factors:
            sections.append(f"SUPPORTING FACTORS:\n" + "\n".join(f"• {factor}" for factor in self.supporting_factors) + "\n")
        
        if self.conflicting_factors:
            sections.append(f"CONFLICTING FACTORS:\n" + "\n".join(f"• {factor}" for factor in self.conflicting_factors) + "\n")
        
        if self.key_warnings:
            sections.append(f"KEY WARNINGS:\n" + "\n".join(f"⚠️ {warning}" for warning in self.key_warnings) + "\n")
        
        return "\n".join(sections)


class NarrativeGenerator:
    """
    Main narrative generation engine.
    
    Converts pattern analysis, multi-timeframe data, and trading signals
    into comprehensive human-readable narratives for LLM consumption.
    """
    
    def __init__(self, config: Optional[NarrativeConfiguration] = None):
        self.config = config or NarrativeConfiguration()
    
    def generate_signal_narrative(self,
                                signal: GeneratedSignal,
                                market_data: Optional[List[CandlestickData]] = None) -> TradingNarrative:
        """Generate comprehensive narrative from trading signal."""
        
        # Extract components
        pattern = signal.source_pattern
        pattern_score = signal.pattern_score
        timeframe_analysis = signal.timeframe_analysis
        entry_point = signal.entry_point
        risk_mgmt = signal.risk_management
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(signal)
        
        # Generate market overview
        market_overview = self._generate_market_overview(pattern, pattern_score, market_data)
        
        # Generate pattern analysis
        pattern_narrative = PatternNarrative.from_pattern(pattern, pattern_score, self.config)
        pattern_analysis = self._compile_pattern_analysis(pattern_narrative)
        
        # Generate timeframe analysis if available
        timeframe_analysis_text = None
        if timeframe_analysis:
            timeframe_narrative = TimeframeNarrative.from_timeframe_analysis(timeframe_analysis, self.config)
            timeframe_analysis_text = self._compile_timeframe_analysis(timeframe_narrative)
        
        # Generate volume analysis
        volume_analysis = self._generate_volume_analysis(pattern_score)
        
        # Generate risk assessment
        risk_assessment = self._generate_risk_assessment(risk_mgmt, pattern_score)
        
        # Generate entry strategy
        entry_strategy = self._generate_entry_strategy(entry_point, pattern)
        
        # Generate exit strategy
        exit_strategy = self._generate_exit_strategy(risk_mgmt, pattern_score)
        
        # Generate confidence rationale
        confidence_rationale = self._generate_confidence_rationale(pattern_score)
        
        # Generate supporting and conflicting factors
        supporting_factors = self._generate_supporting_factors(signal)
        conflicting_factors = self._generate_conflicting_factors(signal)
        
        # Generate warnings
        key_warnings = self._generate_key_warnings(signal)
        
        return TradingNarrative(
            executive_summary=executive_summary,
            market_overview=market_overview,
            pattern_analysis=pattern_analysis,
            timeframe_analysis=timeframe_analysis_text,
            volume_analysis=volume_analysis,
            risk_assessment=risk_assessment,
            entry_strategy=entry_strategy,
            exit_strategy=exit_strategy,
            confidence_rationale=confidence_rationale,
            supporting_factors=supporting_factors,
            conflicting_factors=conflicting_factors,
            key_warnings=key_warnings
        )
    
    def _generate_executive_summary(self, signal: GeneratedSignal) -> str:
        """Generate executive summary."""
        pattern = signal.source_pattern
        pattern_name = pattern.pattern_type.value.replace('_', ' ').title()
        direction = "BULLISH" if pattern.directional_bias in [SignalDirection.BUY, SignalDirection.STRONG_BUY] else "BEARISH"
        confidence = int(signal.pattern_score.weighted_confidence_score)
        symbol = pattern.symbol
        timeframe = PatternNarrative._timeframe_to_text(pattern.timeframe)
        
        return (f"{direction} {pattern_name} pattern detected on {symbol} {timeframe} chart with "
                f"{confidence}% confidence. Pattern suggests {direction.lower()} momentum with "
                f"entry recommended around ${signal.entry_point.entry_price} and "
                f"risk management targeting {signal.risk_management.risk_reward_ratio:.1f}:1 risk/reward.")
    
    def _generate_market_overview(self,
                                pattern: CandlestickPattern,
                                pattern_score: CompositePatternScore,
                                market_data: Optional[List[CandlestickData]]) -> str:
        """Generate market overview."""
        session_score = int(pattern_score.context_scoring.session_score)
        volatility_score = int(pattern_score.context_scoring.volatility_score)
        trend_score = int(pattern_score.context_scoring.trend_alignment_score)
        
        overview = f"Market conditions show "
        
        if session_score >= 80:
            overview += "optimal trading session timing with high liquidity, "
        elif session_score >= 60:
            overview += "favorable session conditions with adequate liquidity, "
        else:
            overview += "sub-optimal session timing with reduced liquidity, "
        
        if volatility_score >= 70:
            overview += "elevated volatility providing good movement potential, "
        elif volatility_score >= 50:
            overview += "normal volatility levels supporting pattern development, "
        else:
            overview += "low volatility potentially limiting price movement, "
        
        if trend_score >= 70:
            overview += "and strong trend alignment supporting the pattern direction."
        elif trend_score >= 50:
            overview += "and reasonable trend context for the pattern."
        else:
            overview += "but challenging trend environment for pattern execution."
        
        return overview
    
    def _compile_pattern_analysis(self, pattern_narrative: PatternNarrative) -> str:
        """Compile pattern analysis from narrative."""
        return (f"{pattern_narrative.description}. {pattern_narrative.significance} "
                f"{pattern_narrative.strength_assessment} {pattern_narrative.reliability_note} "
                f"{pattern_narrative.context_relevance}")
    
    def _compile_timeframe_analysis(self, timeframe_narrative: TimeframeNarrative) -> str:
        """Compile timeframe analysis from narrative."""
        return (f"{timeframe_narrative.confluence_summary} {timeframe_narrative.trend_alignment} "
                f"{timeframe_narrative.strength_distribution} {timeframe_narrative.conflict_analysis} "
                f"{timeframe_narrative.dominant_timeframe} {timeframe_narrative.recommendation_basis}")
    
    def _generate_volume_analysis(self, pattern_score: CompositePatternScore) -> str:
        """Generate volume analysis."""
        volume_score = pattern_score.volume_scoring.overall_volume_score
        spike_score = pattern_score.volume_scoring.volume_spike_score
        trend_score = pattern_score.volume_scoring.volume_trend_score
        
        analysis = f"Volume analysis shows {int(volume_score)}/100 overall confirmation. "
        
        if spike_score >= 70:
            analysis += "Strong volume spikes during pattern formation provide excellent confirmation. "
        elif spike_score >= 50:
            analysis += "Moderate volume activity supports pattern validity. "
        else:
            analysis += "Limited volume confirmation raises caution about pattern strength. "
        
        if trend_score >= 70:
            analysis += "Volume trend strongly aligns with pattern direction."
        elif trend_score >= 50:
            analysis += "Volume trend moderately supports pattern direction."
        else:
            analysis += "Volume trend shows weak alignment with pattern signals."
        
        return analysis
    
    def _generate_risk_assessment(self,
                                risk_mgmt: SignalRiskManagement,
                                pattern_score: CompositePatternScore) -> str:
        """Generate risk assessment."""
        risk_level = risk_mgmt.risk_level.value.replace('_', ' ').title()
        risk_pct = float(risk_mgmt.risk_percentage)
        rr_ratio = float(risk_mgmt.risk_reward_ratio) if risk_mgmt.risk_reward_ratio else 0
        confidence = int(pattern_score.weighted_confidence_score)
        
        assessment = f"{risk_level} risk trade with {risk_pct}% account risk recommended. "
        
        if rr_ratio >= 2.0:
            assessment += f"Excellent {rr_ratio:.1f}:1 risk/reward ratio provides strong profit potential. "
        elif rr_ratio >= 1.5:
            assessment += f"Good {rr_ratio:.1f}:1 risk/reward ratio offers reasonable upside. "
        else:
            assessment += f"Marginal {rr_ratio:.1f}:1 risk/reward ratio requires careful consideration. "
        
        if confidence >= 80:
            assessment += "High confidence level supports increased position sizing within risk limits."
        elif confidence >= 60:
            assessment += "Moderate confidence suggests standard position sizing approach."
        else:
            assessment += "Lower confidence warrants reduced position sizing or waiting for better setups."
        
        return assessment
    
    def _generate_entry_strategy(self,
                               entry_point: SignalEntryPoint,
                               pattern: CandlestickPattern) -> str:
        """Generate entry strategy."""
        timing = entry_point.entry_timing.value.replace('_', ' ')
        entry_price = float(entry_point.entry_price)
        zone_size = float(entry_point.entry_zone_size)
        
        strategy = f"Recommended {timing} entry around ${entry_price:.2f} "
        
        if entry_point.entry_timing == SignalTiming.IMMEDIATE:
            strategy += "as pattern completion provides immediate signal validation. "
        elif entry_point.entry_timing == SignalTiming.CONFIRMATION:
            strategy += "after waiting for next candle confirmation of pattern direction. "
        elif entry_point.entry_timing == SignalTiming.BREAKOUT:
            strategy += "upon price breakout from pattern boundaries with volume confirmation. "
        else:
            strategy += "using appropriate timing based on market conditions. "
        
        strategy += f"Entry zone spans ${zone_size:.2f} providing flexibility for optimal fill."
        
        return strategy
    
    def _generate_exit_strategy(self,
                              risk_mgmt: SignalRiskManagement,
                              pattern_score: CompositePatternScore) -> str:
        """Generate exit strategy."""
        stop_loss = float(risk_mgmt.stop_loss_price)
        take_profit = float(risk_mgmt.take_profit_price) if risk_mgmt.take_profit_price else 0
        confidence = int(pattern_score.weighted_confidence_score)
        
        strategy = f"Stop loss placed at ${stop_loss:.2f} based on pattern structure invalidation. "
        
        if take_profit > 0:
            strategy += f"Initial profit target at ${take_profit:.2f} with potential for extension "
        else:
            strategy += "Profit targets based on technical resistance levels with potential for extension "
        
        if confidence >= 80:
            strategy += "given high pattern confidence and strong technical setup."
        elif confidence >= 60:
            strategy += "considering moderate pattern strength and market conditions."
        else:
            strategy += "with caution due to lower pattern confidence levels."
        
        return strategy
    
    def _generate_confidence_rationale(self, pattern_score: CompositePatternScore) -> str:
        """Generate confidence rationale."""
        confidence = int(pattern_score.weighted_confidence_score)
        technical = int(pattern_score.technical_scoring.overall_technical_score)
        volume = int(pattern_score.volume_scoring.overall_volume_score)
        context = int(pattern_score.context_scoring.overall_context_score)
        historical = int(pattern_score.historical_performance.success_rate)
        
        rationale = f"{confidence}% overall confidence derived from: "
        rationale += f"Technical quality ({technical}/100), "
        rationale += f"Volume confirmation ({volume}/100), "
        rationale += f"Market context ({context}/100), "
        rationale += f"Historical performance ({historical}% success rate). "
        
        if confidence >= 80:
            rationale += "High confidence warrants aggressive position sizing within risk parameters."
        elif confidence >= 60:
            rationale += "Moderate confidence supports standard trading approach."
        else:
            rationale += "Lower confidence suggests conservative position sizing or alternative setups."
        
        return rationale
    
    def _generate_supporting_factors(self, signal: GeneratedSignal) -> List[str]:
        """Generate list of supporting factors."""
        factors = []
        pattern_score = signal.pattern_score
        
        # Technical factors
        if pattern_score.technical_scoring.overall_technical_score >= 75:
            factors.append("Excellent technical pattern structure with strong candlestick characteristics")
        
        # Volume factors
        if pattern_score.volume_scoring.volume_spike_score >= 70:
            factors.append("Strong volume confirmation during pattern formation")
        
        # Context factors
        if pattern_score.context_scoring.session_score >= 80:
            factors.append("Optimal market session timing with high liquidity")
        
        if pattern_score.context_scoring.trend_alignment_score >= 70:
            factors.append("Strong alignment with broader market trend")
        
        # Historical factors
        if pattern_score.historical_performance.success_rate >= 70:
            factors.append(f"Strong historical performance with {int(pattern_score.historical_performance.success_rate)}% success rate")
        
        # Risk factors
        if signal.risk_management.risk_reward_ratio and signal.risk_management.risk_reward_ratio >= 2:
            factors.append(f"Favorable {signal.risk_management.risk_reward_ratio:.1f}:1 risk/reward ratio")
        
        # Multi-timeframe factors
        if signal.timeframe_analysis and signal.timeframe_analysis.confluence_analysis.confluence_score >= 70:
            factors.append("Strong multi-timeframe confluence supporting signal direction")
        
        return factors
    
    def _generate_conflicting_factors(self, signal: GeneratedSignal) -> List[str]:
        """Generate list of conflicting factors."""
        factors = []
        pattern_score = signal.pattern_score
        
        # Technical conflicts
        if pattern_score.technical_scoring.overall_technical_score < 50:
            factors.append("Weak technical pattern structure raises quality concerns")
        
        # Volume conflicts
        if pattern_score.volume_scoring.overall_volume_score < 40:
            factors.append("Limited volume confirmation weakens pattern validity")
        
        # Context conflicts
        if pattern_score.context_scoring.volatility_score < 30:
            factors.append("Low volatility environment may limit price movement potential")
        
        if pattern_score.context_scoring.trend_alignment_score < 40:
            factors.append("Pattern direction conflicts with broader market trend")
        
        # Historical conflicts
        if pattern_score.historical_performance.success_rate < 50:
            factors.append(f"Below-average historical performance with {int(pattern_score.historical_performance.success_rate)}% success rate")
        
        # Risk conflicts
        if signal.risk_management.risk_reward_ratio and signal.risk_management.risk_reward_ratio < 1.5:
            factors.append(f"Suboptimal {signal.risk_management.risk_reward_ratio:.1f}:1 risk/reward ratio")
        
        # Multi-timeframe conflicts
        if signal.timeframe_analysis and signal.timeframe_analysis.trend_alignment.conflicting_timeframes:
            factors.append("Conflicting signals detected across multiple timeframes")
        
        return factors
    
    def _generate_key_warnings(self, signal: GeneratedSignal) -> List[str]:
        """Generate key warnings."""
        warnings = []
        pattern_score = signal.pattern_score
        
        # Confidence warnings
        if pattern_score.weighted_confidence_score < 60:
            warnings.append("Low pattern confidence requires reduced position sizing")
        
        # Volume warnings
        if pattern_score.volume_scoring.overall_volume_score < 40:
            warnings.append("Weak volume confirmation increases risk of false signal")
        
        # Session warnings
        if pattern_score.context_scoring.session_score < 40:
            warnings.append("Off-hours trading session may result in poor liquidity and wider spreads")
        
        # Risk warnings
        if signal.risk_management.risk_level == RiskLevel.VERY_HIGH:
            warnings.append("Very high risk classification requires careful position management")
        
        if signal.risk_management.risk_reward_ratio and signal.risk_management.risk_reward_ratio < 1.2:
            warnings.append("Poor risk/reward ratio makes this trade marginally profitable")
        
        # Historical warnings
        if pattern_score.historical_performance.total_occurrences < 10:
            warnings.append("Limited historical data reduces reliability of performance metrics")
        
        # Expiry warnings
        if signal.base_signal.expiry:
            time_to_expiry = signal.base_signal.expiry - datetime.now(timezone.utc)
            if time_to_expiry.total_seconds() < 3600:  # Less than 1 hour
                warnings.append("Signal expires soon - immediate action required if trading")
        
        return warnings
    
    def generate_pattern_summary(self,
                               patterns: List[CandlestickPattern],
                               pattern_scores: List[CompositePatternScore]) -> str:
        """Generate summary of multiple patterns."""
        if not patterns:
            return "No significant patterns detected in current market analysis."
        
        # Sort patterns by confidence
        sorted_patterns = sorted(
            zip(patterns, pattern_scores),
            key=lambda x: x[1].weighted_confidence_score,
            reverse=True
        )
        
        summary = f"Analysis of {len(patterns)} candlestick patterns:\n\n"
        
        for i, (pattern, score) in enumerate(sorted_patterns[:self.config.max_pattern_count], 1):
            pattern_name = pattern.pattern_type.value.replace('_', ' ').title()
            direction = "bullish" if pattern.directional_bias in [SignalDirection.BUY, SignalDirection.STRONG_BUY] else "bearish"
            confidence = int(score.weighted_confidence_score)
            timeframe = PatternNarrative._timeframe_to_text(pattern.timeframe)
            
            summary += f"{i}. {pattern_name} ({timeframe}): {direction} bias, {confidence}% confidence\n"
            summary += f"   Technical: {int(score.technical_scoring.overall_technical_score)}/100, "
            summary += f"Volume: {int(score.volume_scoring.overall_volume_score)}/100, "
            summary += f"Context: {int(score.context_scoring.overall_context_score)}/100\n\n"
        
        return summary
    
    def generate_quick_narrative(self,
                               signal: GeneratedSignal,
                               style: NarrativeStyle = NarrativeStyle.CONCISE) -> str:
        """Generate quick narrative for immediate consumption."""
        pattern = signal.source_pattern
        pattern_name = pattern.pattern_type.value.replace('_', ' ').title()
        direction = "BULLISH" if pattern.directional_bias in [SignalDirection.BUY, SignalDirection.STRONG_BUY] else "BEARISH"
        confidence = int(signal.pattern_score.weighted_confidence_score)
        entry_price = float(signal.entry_point.entry_price)
        stop_loss = float(signal.risk_management.stop_loss_price)
        rr_ratio = float(signal.risk_management.risk_reward_ratio) if signal.risk_management.risk_reward_ratio else 0
        
        if style == NarrativeStyle.CONCISE:
            return (f"{direction} {pattern_name}: {confidence}% confidence, "
                    f"Entry: ${entry_price:.2f}, Stop: ${stop_loss:.2f}, R/R: {rr_ratio:.1f}:1")
        
        return self.generate_signal_narrative(signal).executive_summary 