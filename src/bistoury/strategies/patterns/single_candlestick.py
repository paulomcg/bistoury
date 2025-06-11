"""
Single Candlestick Pattern Recognition

This module implements recognition algorithms for single-candlestick patterns
including Doji, Hammer, Shooting Star, Spinning Top, Marubozu, and others.

Each pattern detector calculates:
- Pattern confidence based on technical criteria
- Bullish/bearish probability
- Pattern strength and reliability scores
- Volume confirmation when applicable

Built on top of existing CandlestickData and CandlestickPattern models.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Dict, Any, List
from abc import ABC, abstractmethod

from ...models.market_data import CandlestickData, Timeframe
from ...models.signals import CandlestickPattern, PatternType, SignalDirection
from ..candlestick_models import PatternStrength, VolumeProfile
from .pattern_config import get_pattern_config
from .config_manager import initialize_pattern_config

# Initialize pattern configuration on module import
try:
    initialize_pattern_config()
except Exception as e:
    print(f"Warning: Could not initialize pattern configuration: {e}")
    print("Using default hardcoded values")


class SinglePatternDetector(ABC):
    """
    Abstract base class for single candlestick pattern detectors.
    
    Provides common functionality and interface for all single-pattern
    recognition algorithms.
    """
    
    def __init__(self, min_confidence: Decimal = Decimal('60')):
        """Initialize detector with minimum confidence threshold."""
        self.min_confidence = min_confidence
    
    @abstractmethod
    def detect(self, candle: CandlestickData, volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """
        Detect pattern in a single candlestick.
        
        Args:
            candle: Candlestick data to analyze
            volume_profile: Optional volume profile for confirmation
            
        Returns:
            CandlestickPattern if detected, None otherwise
        """
        pass
    
    @abstractmethod
    def get_pattern_type(self) -> PatternType:
        """Get the pattern type this detector recognizes."""
        pass
    
    def _calculate_body_to_range_ratio(self, candle: CandlestickData) -> Decimal:
        """Calculate the ratio of body size to total range."""
        total_range = candle.high - candle.low
        if total_range == 0:
            return Decimal('0')
        return candle.body_size / total_range
    
    def _calculate_upper_shadow_ratio(self, candle: CandlestickData) -> Decimal:
        """Calculate upper shadow as ratio of total range."""
        total_range = candle.high - candle.low
        if total_range == 0:
            return Decimal('0')
        return candle.upper_shadow / total_range
    
    def _calculate_lower_shadow_ratio(self, candle: CandlestickData) -> Decimal:
        """Calculate lower shadow as ratio of total range."""
        total_range = candle.high - candle.low
        if total_range == 0:
            return Decimal('0')
        return candle.lower_shadow / total_range
    
    def _create_pattern(
        self,
        candle: CandlestickData,
        confidence: Decimal,
        reliability: Decimal,
        bullish_probability: Decimal,
        bearish_probability: Decimal,
        volume_profile: Optional[VolumeProfile] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CandlestickPattern:
        """Create a CandlestickPattern instance with standard fields."""
        pattern_id = f"{self.get_pattern_type().value}_{candle.symbol}_{int(candle.timestamp.timestamp() * 1000)}"
        
        return CandlestickPattern(
            pattern_id=pattern_id,
            symbol=candle.symbol,
            pattern_type=self.get_pattern_type(),
            candles=[candle],
            timeframe=candle.timeframe,
            confidence=confidence,
            reliability=reliability,
            bullish_probability=bullish_probability,
            bearish_probability=bearish_probability,
            completion_price=candle.close,
            volume_confirmation=volume_profile.volume_confirmation if volume_profile else False,
            pattern_metadata=metadata or {}
        )


class DojiDetector(SinglePatternDetector):
    """
    Doji pattern detector.
    
    A Doji occurs when open and close prices are very close or equal,
    indicating market indecision. Variations include:
    - Standard Doji: Small body with upper and lower shadows
    - Long-legged Doji: Very long shadows relative to body
    - Dragonfly Doji: Long lower shadow, no upper shadow
    - Gravestone Doji: Long upper shadow, no lower shadow
    """
    
    def get_pattern_type(self) -> PatternType:
        return PatternType.DOJI
    
    def detect(self, candle: CandlestickData, volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Detect Doji pattern."""
        config = get_pattern_config().doji
        
        body_ratio = self._calculate_body_to_range_ratio(candle)
        upper_shadow_ratio = self._calculate_upper_shadow_ratio(candle)
        lower_shadow_ratio = self._calculate_lower_shadow_ratio(candle)
        
        # Doji requires very small body relative to range
        if body_ratio > config.max_body_ratio:
            return None
        
        # Also require meaningful range (avoid noise on flat markets)
        total_range = candle.high - candle.low
        range_ratio = total_range / candle.close
        if range_ratio < config.min_range_ratio:
            return None
        
        # Calculate confidence based on how small the body is
        body_score = (config.max_body_ratio - body_ratio) / config.max_body_ratio * config.body_score_weight
        
        # Bonus for symmetric shadows (more indecision)
        shadow_symmetry = Decimal('1') - abs(upper_shadow_ratio - lower_shadow_ratio)
        symmetry_bonus = shadow_symmetry * config.symmetry_bonus_weight
        
        # Bonus for good range (meaningful price movement)
        range_bonus = min(config.range_bonus_weight, range_ratio * config.range_bonus_multiplier)
        
        confidence = min(Decimal('100'), body_score + symmetry_bonus + range_bonus)
        
        if confidence < self.min_confidence:
            return None
        
        # Determine Doji subtype
        doji_subtype = self._classify_doji_subtype(upper_shadow_ratio, lower_shadow_ratio, config)
        
        # Doji is neutral but can be bullish/bearish based on context
        if doji_subtype == "dragonfly":
            bullish_prob = config.dragonfly_bullish_prob
            bearish_prob = config.dragonfly_bearish_prob
        elif doji_subtype == "gravestone":
            bullish_prob = config.gravestone_bullish_prob
            bearish_prob = config.gravestone_bearish_prob
        else:
            bullish_prob = config.standard_bullish_prob
            bearish_prob = config.standard_bearish_prob
        
        # Higher reliability for more perfect Doji
        reliability = config.base_reliability + (symmetry_bonus / Decimal('100'))
        
        metadata = {
            "doji_subtype": doji_subtype,
            "body_ratio": float(body_ratio),
            "upper_shadow_ratio": float(upper_shadow_ratio),
            "lower_shadow_ratio": float(lower_shadow_ratio),
            "shadow_symmetry": float(shadow_symmetry)
        }
        
        return self._create_pattern(
            candle=candle,
            confidence=confidence,
            reliability=reliability,
            bullish_probability=bullish_prob,
            bearish_probability=bearish_prob,
            volume_profile=volume_profile,
            metadata=metadata
        )
    
    def _classify_doji_subtype(self, upper_ratio: Decimal, lower_ratio: Decimal, config) -> str:
        """Classify the type of Doji based on shadow ratios."""
        if lower_ratio >= config.min_long_shadow and upper_ratio <= config.max_short_shadow:
            return "dragonfly"
        elif upper_ratio >= config.min_long_shadow and lower_ratio <= config.max_short_shadow:
            return "gravestone"
        elif upper_ratio >= config.min_long_shadow and lower_ratio >= config.min_long_shadow:
            return "long_legged"
        else:
            return "standard"


class HammerDetector(SinglePatternDetector):
    """
    Hammer pattern detector.
    
    A Hammer is a bullish reversal pattern characterized by:
    - Small body near the top of the range
    - Long lower shadow (at least 2x body size)
    - Little or no upper shadow
    - Can be bullish (green) or bearish (red) body
    """
    
    def get_pattern_type(self) -> PatternType:
        return PatternType.HAMMER
    
    def detect(self, candle: CandlestickData, volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Detect Hammer pattern."""
        config = get_pattern_config().hammer
        
        body_ratio = self._calculate_body_to_range_ratio(candle)
        upper_shadow_ratio = self._calculate_upper_shadow_ratio(candle)
        lower_shadow_ratio = self._calculate_lower_shadow_ratio(candle)
        
        # Hammer requirements
        if (lower_shadow_ratio < config.min_lower_shadow or 
            upper_shadow_ratio > config.max_upper_shadow or 
            body_ratio > config.max_body_ratio):
            return None
        
        # Calculate confidence based on how well it fits criteria
        lower_shadow_score = min(config.max_lower_shadow_score, (lower_shadow_ratio - config.min_lower_shadow) * config.lower_shadow_score_multiplier)
        upper_shadow_score = max(Decimal('0'), (config.max_upper_shadow - upper_shadow_ratio) * config.upper_shadow_score_multiplier)
        body_score = (config.max_body_ratio - body_ratio) / config.max_body_ratio * config.body_score_weight
        
        confidence = config.base_confidence + lower_shadow_score + upper_shadow_score + body_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Hammer is typically bullish reversal
        # Bullish body (close > open) gives slight edge
        if candle.is_bullish:
            bullish_prob = config.bullish_body_bullish_prob
            bearish_prob = config.bullish_body_bearish_prob
        else:
            bullish_prob = config.bearish_body_bullish_prob
            bearish_prob = config.bearish_body_bearish_prob
        
        # Reliability based on how textbook the pattern is
        reliability = config.base_reliability + (confidence - Decimal('60')) / config.reliability_divisor
        reliability = min(config.max_reliability, max(config.min_reliability, reliability))
        
        metadata = {
            "body_ratio": float(body_ratio),
            "upper_shadow_ratio": float(upper_shadow_ratio),
            "lower_shadow_ratio": float(lower_shadow_ratio),
            "is_bullish_body": candle.is_bullish,
            "lower_shadow_to_body": float(candle.lower_shadow / candle.body_size) if candle.body_size > 0 else None
        }
        
        return self._create_pattern(
            candle=candle,
            confidence=confidence,
            reliability=reliability,
            bullish_probability=bullish_prob,
            bearish_probability=bearish_prob,
            volume_profile=volume_profile,
            metadata=metadata
        )


class ShootingStarDetector(SinglePatternDetector):
    """
    Shooting Star pattern detector.
    
    A Shooting Star is a bearish reversal pattern characterized by:
    - Small body near the bottom of the range
    - Long upper shadow (at least 2x body size)
    - Little or no lower shadow
    - Can be bullish (green) or bearish (red) body
    """
    
    def get_pattern_type(self) -> PatternType:
        return PatternType.SHOOTING_STAR
    
    def detect(self, candle: CandlestickData, volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Detect Shooting Star pattern."""
        body_ratio = self._calculate_body_to_range_ratio(candle)
        upper_shadow_ratio = self._calculate_upper_shadow_ratio(candle)
        lower_shadow_ratio = self._calculate_lower_shadow_ratio(candle)
        
        config = get_pattern_config().shooting_star
        
        # Shooting Star requirements (inverse of Hammer)
        if (upper_shadow_ratio < config.min_upper_shadow or 
            lower_shadow_ratio > config.max_lower_shadow or 
            body_ratio > config.max_body_ratio):
            return None
        
        # Calculate confidence (similar to Hammer but for upper shadow)
        upper_shadow_score = min(config.max_upper_shadow_score, (upper_shadow_ratio - config.min_upper_shadow) * config.upper_shadow_score_multiplier)
        lower_shadow_score = max(Decimal('0'), (config.max_lower_shadow - lower_shadow_ratio) * config.lower_shadow_score_multiplier)
        body_score = (config.max_body_ratio - body_ratio) / config.max_body_ratio * config.body_score_weight
        
        confidence = config.base_confidence + upper_shadow_score + lower_shadow_score + body_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Shooting Star is typically bearish reversal
        # Bearish body (close < open) gives slight edge
        if candle.is_bearish:
            bullish_prob = config.bearish_body_bullish_prob
            bearish_prob = config.bearish_body_bearish_prob
        else:
            bullish_prob = config.bullish_body_bullish_prob
            bearish_prob = config.bullish_body_bearish_prob
        
        # Reliability based on pattern quality
        reliability = config.base_reliability + (confidence - Decimal('60')) / config.reliability_divisor
        reliability = min(config.max_reliability, max(config.min_reliability, reliability))
        
        metadata = {
            "body_ratio": float(body_ratio),
            "upper_shadow_ratio": float(upper_shadow_ratio),
            "lower_shadow_ratio": float(lower_shadow_ratio),
            "is_bearish_body": candle.is_bearish,
            "upper_shadow_to_body": float(candle.upper_shadow / candle.body_size) if candle.body_size > 0 else None
        }
        
        return self._create_pattern(
            candle=candle,
            confidence=confidence,
            reliability=reliability,
            bullish_probability=bullish_prob,
            bearish_probability=bearish_prob,
            volume_profile=volume_profile,
            metadata=metadata
        )


class SpinningTopDetector(SinglePatternDetector):
    """
    Spinning Top pattern detector.
    
    A Spinning Top indicates indecision with:
    - Small body relative to shadows
    - Upper and lower shadows of similar length
    - Body can be bullish or bearish
    - Indicates potential trend reversal or continuation
    """
    
    def get_pattern_type(self) -> PatternType:
        return PatternType.SPINNING_TOP
    
    def detect(self, candle: CandlestickData, volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Detect Spinning Top pattern."""
        config = get_pattern_config().spinning_top
        
        body_ratio = self._calculate_body_to_range_ratio(candle)
        upper_shadow_ratio = self._calculate_upper_shadow_ratio(candle)
        lower_shadow_ratio = self._calculate_lower_shadow_ratio(candle)
        
        shadow_difference = abs(upper_shadow_ratio - lower_shadow_ratio)
        
        if (body_ratio > config.max_body_ratio or 
            upper_shadow_ratio < config.min_shadow_ratio or 
            lower_shadow_ratio < config.min_shadow_ratio or
            shadow_difference > config.max_shadow_diff):
            return None
        
        # Calculate confidence based on shadow balance and body size
        shadow_balance_score = (config.max_shadow_diff - shadow_difference) / config.max_shadow_diff * config.shadow_balance_weight
        body_score = (config.max_body_ratio - body_ratio) / config.max_body_ratio * config.body_score_weight
        shadow_length_score = min(config.shadow_length_weight, (upper_shadow_ratio + lower_shadow_ratio) * config.shadow_length_multiplier)
        
        confidence = shadow_balance_score + body_score + shadow_length_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Spinning Top is neutral with slight bias based on body color
        if candle.is_bullish:
            bullish_prob = config.bullish_body_bullish_prob
            bearish_prob = config.bullish_body_bearish_prob
        else:
            bullish_prob = config.bearish_body_bullish_prob
            bearish_prob = config.bearish_body_bearish_prob
        
        # Moderate reliability - pattern indicates indecision
        reliability = config.base_reliability + (confidence - Decimal('60')) / config.reliability_divisor
        reliability = min(config.max_reliability, max(config.min_reliability, reliability))
        
        metadata = {
            "body_ratio": float(body_ratio),
            "upper_shadow_ratio": float(upper_shadow_ratio),
            "lower_shadow_ratio": float(lower_shadow_ratio),
            "shadow_difference": float(shadow_difference),
            "shadow_balance": float(shadow_balance_score / Decimal('40'))
        }
        
        return self._create_pattern(
            candle=candle,
            confidence=confidence,
            reliability=reliability,
            bullish_probability=bullish_prob,
            bearish_probability=bearish_prob,
            volume_profile=volume_profile,
            metadata=metadata
        )


class MarubozuDetector(SinglePatternDetector):
    """
    Marubozu pattern detector.
    
    A Marubozu is a strong directional candlestick with:
    - Large body with little to no shadows
    - Open equals low (bullish) or high (bearish)
    - Close equals high (bullish) or low (bearish)
    - Indicates strong momentum in one direction
    """
    
    def get_pattern_type(self) -> PatternType:
        return PatternType.MARUBOZU
    
    def detect(self, candle: CandlestickData, volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Detect Marubozu pattern."""
        config = get_pattern_config().marubozu
        
        body_ratio = self._calculate_body_to_range_ratio(candle)
        upper_shadow_ratio = self._calculate_upper_shadow_ratio(candle)
        lower_shadow_ratio = self._calculate_lower_shadow_ratio(candle)
        
        # Marubozu requirements
        if (body_ratio < config.min_body_ratio or 
            upper_shadow_ratio > config.max_shadow_ratio or 
            lower_shadow_ratio > config.max_shadow_ratio):
            return None
        
        # Calculate confidence based on body size and lack of shadows
        body_score = (body_ratio - config.min_body_ratio) / (Decimal('1') - config.min_body_ratio) * config.body_score_weight
        shadow_score = (config.max_shadow_ratio - max(upper_shadow_ratio, lower_shadow_ratio)) / config.max_shadow_ratio * config.shadow_score_weight
        
        confidence = body_score + shadow_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Marubozu strongly indicates continuation of current direction
        if candle.is_bullish:
            bullish_prob = config.bullish_bullish_prob
            bearish_prob = config.bullish_bearish_prob
        else:
            bullish_prob = config.bearish_bullish_prob
            bearish_prob = config.bearish_bearish_prob
        
        # High reliability for strong directional patterns
        reliability = config.base_reliability + (confidence - Decimal('60')) / config.reliability_divisor
        reliability = min(config.max_reliability, max(config.min_reliability, reliability))
        
        metadata = {
            "body_ratio": float(body_ratio),
            "upper_shadow_ratio": float(upper_shadow_ratio),
            "lower_shadow_ratio": float(lower_shadow_ratio),
            "is_bullish": candle.is_bullish,
            "marubozu_type": "bullish" if candle.is_bullish else "bearish"
        }
        
        return self._create_pattern(
            candle=candle,
            confidence=confidence,
            reliability=reliability,
            bullish_probability=bullish_prob,
            bearish_probability=bearish_prob,
            volume_profile=volume_profile,
            metadata=metadata
        )


class SinglePatternRecognizer:
    """
    Main class for single candlestick pattern recognition.
    
    Coordinates multiple pattern detectors and returns all detected patterns
    with their confidence scores and classifications.
    """
    
    def __init__(self, min_confidence: Decimal = Decimal('60')):
        """Initialize with all single pattern detectors."""
        self.min_confidence = min_confidence
        self.detectors = [
            DojiDetector(min_confidence),
            HammerDetector(min_confidence),
            ShootingStarDetector(min_confidence),
            SpinningTopDetector(min_confidence),
            MarubozuDetector(min_confidence)
        ]
    
    def analyze(self, candle: CandlestickData, volume_profile: Optional[VolumeProfile] = None) -> List[CandlestickPattern]:
        """
        Analyze a candlestick for all single patterns.
        
        Args:
            candle: Candlestick data to analyze
            volume_profile: Optional volume profile for confirmation
            
        Returns:
            List of detected patterns sorted by confidence
        """
        patterns = []
        
        for detector in self.detectors:
            try:
                pattern = detector.detect(candle, volume_profile)
                if pattern is not None:
                    patterns.append(pattern)
            except Exception as e:
                # Log error but continue with other detectors
                print(f"Error in {detector.__class__.__name__}: {e}")
                continue
        
        # Sort by confidence (highest first)
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        
        return patterns
    
    def get_best_pattern(self, candle: CandlestickData, volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """
        Get the highest confidence pattern from analysis.
        
        Args:
            candle: Candlestick data to analyze
            volume_profile: Optional volume profile for confirmation
            
        Returns:
            Highest confidence pattern or None if no patterns detected
        """
        patterns = self.analyze(candle, volume_profile)
        return patterns[0] if patterns else None
    
    def get_pattern_summary(self, candle: CandlestickData, volume_profile: Optional[VolumeProfile] = None) -> Dict[str, Any]:
        """
        Get summary statistics for all detected patterns.
        
        Args:
            candle: Candlestick data to analyze
            volume_profile: Optional volume profile for confirmation
            
        Returns:
            Summary dictionary with pattern statistics
        """
        patterns = self.analyze(candle, volume_profile)
        
        if not patterns:
            return {
                "total_patterns": 0,
                "best_pattern": None,
                "avg_confidence": Decimal('0'),
                "bullish_patterns": 0,
                "bearish_patterns": 0,
                "neutral_patterns": 0,
                "pattern_types": []
            }
        
        bullish_count = sum(1 for p in patterns if p.directional_bias == SignalDirection.BUY)
        bearish_count = sum(1 for p in patterns if p.directional_bias == SignalDirection.SELL)
        neutral_count = len(patterns) - bullish_count - bearish_count
        
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
        
        return {
            "total_patterns": len(patterns),
            "best_pattern": patterns[0].pattern_type.value,
            "best_confidence": patterns[0].confidence,
            "avg_confidence": avg_confidence,
            "bullish_patterns": bullish_count,
            "bearish_patterns": bearish_count,
            "neutral_patterns": neutral_count,
            "pattern_types": [p.pattern_type.value for p in patterns]
        } 