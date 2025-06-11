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
from ...config_manager import get_config_manager


class SinglePatternDetector(ABC):
    """
    Abstract base class for single candlestick pattern detectors.
    
    Provides common functionality and interface for all single-pattern
    recognition algorithms.
    """
    
    def __init__(self, min_confidence: Optional[Decimal] = None):
        """
        Initialize detector with minimum confidence threshold.
        
        Args:
            min_confidence: Minimum confidence threshold, defaults to config value
        """
        config_manager = get_config_manager()
        self.min_confidence = min_confidence or config_manager.get_decimal(
            'pattern_detection', 'default_min_confidence', default='20'
        )
    
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
        config_manager = get_config_manager()
        
        # Get configuration values
        max_body_ratio = config_manager.get_decimal('pattern_detection', 'doji', 'max_body_ratio', default='0.02')
        min_range_ratio = config_manager.get_decimal('pattern_detection', 'doji', 'min_range_ratio', default='0.005')
        body_score_weight = config_manager.get_decimal('pattern_detection', 'doji', 'body_score_weight', default='60')
        symmetry_bonus_weight = config_manager.get_decimal('pattern_detection', 'doji', 'symmetry_bonus_weight', default='20')
        range_bonus_weight = config_manager.get_decimal('pattern_detection', 'doji', 'range_bonus_weight', default='20')
        range_bonus_multiplier = config_manager.get_decimal('pattern_detection', 'doji', 'range_bonus_multiplier', default='2000')
        base_reliability = config_manager.get_decimal('pattern_detection', 'doji', 'base_reliability', default='0.7')
        
        body_ratio = self._calculate_body_to_range_ratio(candle)
        upper_shadow_ratio = self._calculate_upper_shadow_ratio(candle)
        lower_shadow_ratio = self._calculate_lower_shadow_ratio(candle)
        
        # Doji requires very small body relative to range
        if body_ratio > max_body_ratio:
            return None
        
        # Also require meaningful range (avoid noise on flat markets)
        total_range = candle.high - candle.low
        range_ratio = total_range / candle.close
        if range_ratio < min_range_ratio:
            return None
        
        # Calculate confidence based on how small the body is
        body_score = (max_body_ratio - body_ratio) / max_body_ratio * body_score_weight
        
        # Bonus for symmetric shadows (more indecision)
        shadow_symmetry = Decimal('1') - abs(upper_shadow_ratio - lower_shadow_ratio)
        symmetry_bonus = shadow_symmetry * symmetry_bonus_weight
        
        # Bonus for good range (meaningful price movement)
        range_bonus = min(range_bonus_weight, range_ratio * range_bonus_multiplier)
        
        confidence = min(Decimal('100'), body_score + symmetry_bonus + range_bonus)
        
        if confidence < self.min_confidence:
            return None
        
        # Determine Doji subtype
        doji_subtype = self._classify_doji_subtype(upper_shadow_ratio, lower_shadow_ratio, config_manager)
        
        # Get probabilities from config
        if doji_subtype == "dragonfly":
            bullish_prob = config_manager.get_decimal('pattern_detection', 'doji', 'dragonfly_bullish_prob', default='0.65')
            bearish_prob = config_manager.get_decimal('pattern_detection', 'doji', 'dragonfly_bearish_prob', default='0.35')
        elif doji_subtype == "gravestone":
            bullish_prob = config_manager.get_decimal('pattern_detection', 'doji', 'gravestone_bullish_prob', default='0.35')
            bearish_prob = config_manager.get_decimal('pattern_detection', 'doji', 'gravestone_bearish_prob', default='0.65')
        else:
            bullish_prob = config_manager.get_decimal('pattern_detection', 'doji', 'standard_bullish_prob', default='0.45')
            bearish_prob = config_manager.get_decimal('pattern_detection', 'doji', 'standard_bearish_prob', default='0.45')
        
        # Higher reliability for more perfect Doji
        reliability = base_reliability + (symmetry_bonus / Decimal('100'))
        
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
    
    def _classify_doji_subtype(self, upper_ratio: Decimal, lower_ratio: Decimal, config_manager) -> str:
        """Classify the type of Doji based on shadow ratios."""
        min_long_shadow = config_manager.get_decimal('pattern_detection', 'doji', 'min_long_shadow', default='0.4')
        max_short_shadow = config_manager.get_decimal('pattern_detection', 'doji', 'max_short_shadow', default='0.1')
        min_long_legged_shadow = config_manager.get_decimal('pattern_detection', 'doji', 'min_long_legged_shadow', default='0.3')
        
        if lower_ratio >= min_long_shadow and upper_ratio <= max_short_shadow:
            return "dragonfly"
        elif upper_ratio >= min_long_shadow and lower_ratio <= max_short_shadow:
            return "gravestone"
        elif upper_ratio >= min_long_legged_shadow and lower_ratio >= min_long_legged_shadow:
            return "long_legged"
        else:
            return "standard"


class HammerDetector(SinglePatternDetector):
    """
    Hammer pattern detector.
    
    A Hammer is a bullish reversal pattern with:
    - Long lower shadow (at least 2x body)
    - Small real body at top of range
    - Little to no upper shadow
    """
    
    def get_pattern_type(self) -> PatternType:
        return PatternType.HAMMER
    
    def detect(self, candle: CandlestickData, volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Detect Hammer pattern."""
        config_manager = get_config_manager()
        
        # Get configuration values
        min_lower_shadow = config_manager.get_decimal('pattern_detection', 'hammer', 'min_lower_shadow', default='0.6')
        max_upper_shadow = config_manager.get_decimal('pattern_detection', 'hammer', 'max_upper_shadow', default='0.15')
        max_body_ratio = config_manager.get_decimal('pattern_detection', 'hammer', 'max_body_ratio', default='0.3')
        base_confidence = config_manager.get_decimal('pattern_detection', 'hammer', 'base_confidence', default='30')
        lower_shadow_score_multiplier = config_manager.get_decimal('pattern_detection', 'hammer', 'lower_shadow_score_multiplier', default='100')
        upper_shadow_score_multiplier = config_manager.get_decimal('pattern_detection', 'hammer', 'upper_shadow_score_multiplier', default='200')
        body_score_weight = config_manager.get_decimal('pattern_detection', 'hammer', 'body_score_weight', default='30')
        max_lower_shadow_score = config_manager.get_decimal('pattern_detection', 'hammer', 'max_lower_shadow_score', default='40')
        base_reliability = config_manager.get_decimal('pattern_detection', 'hammer', 'base_reliability', default='0.75')
        reliability_divisor = config_manager.get_decimal('pattern_detection', 'hammer', 'reliability_divisor', default='200')
        
        body_ratio = self._calculate_body_to_range_ratio(candle)
        upper_shadow_ratio = self._calculate_upper_shadow_ratio(candle)
        lower_shadow_ratio = self._calculate_lower_shadow_ratio(candle)
        
        # Hammer requires long lower shadow and small body
        if lower_shadow_ratio < min_lower_shadow:
            return None
        if upper_shadow_ratio > max_upper_shadow:
            return None
        if body_ratio > max_body_ratio:
            return None
        
        # Calculate confidence
        confidence = base_confidence
        
        # Score based on lower shadow length
        lower_shadow_score = min(max_lower_shadow_score, (lower_shadow_ratio - min_lower_shadow) * lower_shadow_score_multiplier)
        confidence += lower_shadow_score
        
        # Penalty for upper shadow
        upper_shadow_score = max(Decimal('0'), (max_upper_shadow - upper_shadow_ratio) * upper_shadow_score_multiplier)
        confidence += upper_shadow_score
        
        # Score based on small body
        body_score = (max_body_ratio - body_ratio) / max_body_ratio * body_score_weight
        confidence += body_score
        
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Get probabilities based on body color
        if candle.is_bullish:
            bullish_prob = config_manager.get_decimal('pattern_detection', 'hammer', 'bullish_body_bullish_prob', default='0.75')
            bearish_prob = config_manager.get_decimal('pattern_detection', 'hammer', 'bullish_body_bearish_prob', default='0.25')
        else:
            bullish_prob = config_manager.get_decimal('pattern_detection', 'hammer', 'bearish_body_bullish_prob', default='0.70')
            bearish_prob = config_manager.get_decimal('pattern_detection', 'hammer', 'bearish_body_bearish_prob', default='0.30')
        
        # Calculate reliability
        reliability = base_reliability + (confidence - Decimal('60')) / reliability_divisor
        
        metadata = {
            "is_bullish_body": candle.is_bullish,
            "body_ratio": float(body_ratio),
            "upper_shadow_ratio": float(upper_shadow_ratio),
            "lower_shadow_ratio": float(lower_shadow_ratio),
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
        
        config_manager = get_config_manager()
        
        # Shooting Star requirements (inverse of Hammer)
        if (upper_shadow_ratio < config_manager.get_decimal('pattern_detection', 'shooting_star', 'min_upper_shadow', default='0.1') or 
            lower_shadow_ratio > config_manager.get_decimal('pattern_detection', 'shooting_star', 'max_lower_shadow', default='0.2') or 
            body_ratio > config_manager.get_decimal('pattern_detection', 'shooting_star', 'max_body_ratio', default='0.3')):
            return None
        
        # Calculate confidence (similar to Hammer but for upper shadow)
        upper_shadow_score = min(config_manager.get_decimal('pattern_detection', 'shooting_star', 'max_upper_shadow_score', default='40'), (upper_shadow_ratio - config_manager.get_decimal('pattern_detection', 'shooting_star', 'min_upper_shadow', default='0.1')) * config_manager.get_decimal('pattern_detection', 'shooting_star', 'upper_shadow_score_multiplier', default='200'))
        lower_shadow_score = max(Decimal('0'), (config_manager.get_decimal('pattern_detection', 'shooting_star', 'max_lower_shadow', default='0.2') - lower_shadow_ratio) * config_manager.get_decimal('pattern_detection', 'shooting_star', 'lower_shadow_score_multiplier', default='200'))
        body_score = (config_manager.get_decimal('pattern_detection', 'shooting_star', 'max_body_ratio', default='0.3') - body_ratio) / config_manager.get_decimal('pattern_detection', 'shooting_star', 'max_body_ratio', default='0.3') * config_manager.get_decimal('pattern_detection', 'shooting_star', 'body_score_weight', default='30')
        
        confidence = config_manager.get_decimal('pattern_detection', 'shooting_star', 'base_confidence', default='30') + upper_shadow_score + lower_shadow_score + body_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Shooting Star is typically bearish reversal
        # Bearish body (close < open) gives slight edge
        if candle.is_bearish:
            bullish_prob = config_manager.get_decimal('pattern_detection', 'shooting_star', 'bearish_body_bullish_prob', default='0.70')
            bearish_prob = config_manager.get_decimal('pattern_detection', 'shooting_star', 'bearish_body_bearish_prob', default='0.30')
        else:
            bullish_prob = config_manager.get_decimal('pattern_detection', 'shooting_star', 'bullish_body_bullish_prob', default='0.75')
            bearish_prob = config_manager.get_decimal('pattern_detection', 'shooting_star', 'bullish_body_bearish_prob', default='0.25')
        
        # Reliability based on pattern quality
        reliability = config_manager.get_decimal('pattern_detection', 'shooting_star', 'base_reliability', default='0.75') + (confidence - Decimal('60')) / config_manager.get_decimal('pattern_detection', 'shooting_star', 'reliability_divisor', default='200')
        reliability = min(config_manager.get_decimal('pattern_detection', 'shooting_star', 'max_reliability', default='1.0'), max(config_manager.get_decimal('pattern_detection', 'shooting_star', 'min_reliability', default='0.0'), reliability))
        
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
        config_manager = get_config_manager()
        
        body_ratio = self._calculate_body_to_range_ratio(candle)
        upper_shadow_ratio = self._calculate_upper_shadow_ratio(candle)
        lower_shadow_ratio = self._calculate_lower_shadow_ratio(candle)
        
        shadow_difference = abs(upper_shadow_ratio - lower_shadow_ratio)
        
        if (body_ratio > config_manager.get_decimal('pattern_detection', 'spinning_top', 'max_body_ratio', default='0.3') or 
            upper_shadow_ratio < config_manager.get_decimal('pattern_detection', 'spinning_top', 'min_shadow_ratio', default='0.3') or 
            lower_shadow_ratio < config_manager.get_decimal('pattern_detection', 'spinning_top', 'min_shadow_ratio', default='0.3') or
            shadow_difference > config_manager.get_decimal('pattern_detection', 'spinning_top', 'max_shadow_diff', default='0.2')):
            return None
        
        # Calculate confidence based on shadow balance and body size
        shadow_balance_score = (config_manager.get_decimal('pattern_detection', 'spinning_top', 'max_shadow_diff', default='0.2') - shadow_difference) / config_manager.get_decimal('pattern_detection', 'spinning_top', 'max_shadow_diff', default='0.2') * config_manager.get_decimal('pattern_detection', 'spinning_top', 'shadow_balance_weight', default='40')
        body_score = (config_manager.get_decimal('pattern_detection', 'spinning_top', 'max_body_ratio', default='0.3') - body_ratio) / config_manager.get_decimal('pattern_detection', 'spinning_top', 'max_body_ratio', default='0.3') * config_manager.get_decimal('pattern_detection', 'spinning_top', 'body_score_weight', default='30')
        shadow_length_score = min(config_manager.get_decimal('pattern_detection', 'spinning_top', 'shadow_length_weight', default='20'), (upper_shadow_ratio + lower_shadow_ratio) * config_manager.get_decimal('pattern_detection', 'spinning_top', 'shadow_length_multiplier', default='200'))
        
        confidence = shadow_balance_score + body_score + shadow_length_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Spinning Top is neutral with slight bias based on body color
        if candle.is_bullish:
            bullish_prob = config_manager.get_decimal('pattern_detection', 'spinning_top', 'bullish_body_bullish_prob', default='0.75')
            bearish_prob = config_manager.get_decimal('pattern_detection', 'spinning_top', 'bullish_body_bearish_prob', default='0.25')
        else:
            bullish_prob = config_manager.get_decimal('pattern_detection', 'spinning_top', 'bearish_body_bullish_prob', default='0.70')
            bearish_prob = config_manager.get_decimal('pattern_detection', 'spinning_top', 'bearish_body_bearish_prob', default='0.30')
        
        # Moderate reliability - pattern indicates indecision
        reliability = config_manager.get_decimal('pattern_detection', 'spinning_top', 'base_reliability', default='0.75') + (confidence - Decimal('60')) / config_manager.get_decimal('pattern_detection', 'spinning_top', 'reliability_divisor', default='200')
        reliability = min(config_manager.get_decimal('pattern_detection', 'spinning_top', 'max_reliability', default='1.0'), max(config_manager.get_decimal('pattern_detection', 'spinning_top', 'min_reliability', default='0.0'), reliability))
        
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
        config_manager = get_config_manager()
        
        body_ratio = self._calculate_body_to_range_ratio(candle)
        upper_shadow_ratio = self._calculate_upper_shadow_ratio(candle)
        lower_shadow_ratio = self._calculate_lower_shadow_ratio(candle)
        
        # Marubozu requirements
        if (body_ratio < config_manager.get_decimal('pattern_detection', 'marubozu', 'min_body_ratio', default='0.5') or 
            upper_shadow_ratio > config_manager.get_decimal('pattern_detection', 'marubozu', 'max_shadow_ratio', default='0.3') or 
            lower_shadow_ratio > config_manager.get_decimal('pattern_detection', 'marubozu', 'max_shadow_ratio', default='0.3')):
            return None
        
        # Calculate confidence based on body size and lack of shadows
        body_score = (body_ratio - config_manager.get_decimal('pattern_detection', 'marubozu', 'min_body_ratio', default='0.5')) / (Decimal('1') - config_manager.get_decimal('pattern_detection', 'marubozu', 'min_body_ratio', default='0.5')) * config_manager.get_decimal('pattern_detection', 'marubozu', 'body_score_weight', default='30')
        shadow_score = (config_manager.get_decimal('pattern_detection', 'marubozu', 'max_shadow_ratio', default='0.3') - max(upper_shadow_ratio, lower_shadow_ratio)) / config_manager.get_decimal('pattern_detection', 'marubozu', 'max_shadow_ratio', default='0.3') * config_manager.get_decimal('pattern_detection', 'marubozu', 'shadow_score_weight', default='20')
        
        confidence = body_score + shadow_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Marubozu strongly indicates continuation of current direction
        if candle.is_bullish:
            bullish_prob = config_manager.get_decimal('pattern_detection', 'marubozu', 'bullish_bullish_prob', default='0.75')
            bearish_prob = config_manager.get_decimal('pattern_detection', 'marubozu', 'bullish_bearish_prob', default='0.25')
        else:
            bullish_prob = config_manager.get_decimal('pattern_detection', 'marubozu', 'bearish_bullish_prob', default='0.70')
            bearish_prob = config_manager.get_decimal('pattern_detection', 'marubozu', 'bearish_bearish_prob', default='0.30')
        
        # High reliability for strong directional patterns
        reliability = config_manager.get_decimal('pattern_detection', 'marubozu', 'base_reliability', default='0.75') + (confidence - Decimal('60')) / config_manager.get_decimal('pattern_detection', 'marubozu', 'reliability_divisor', default='200')
        reliability = min(config_manager.get_decimal('pattern_detection', 'marubozu', 'max_reliability', default='1.0'), max(config_manager.get_decimal('pattern_detection', 'marubozu', 'min_reliability', default='0.0'), reliability))
        
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
    
    def __init__(self, min_confidence: Decimal = Decimal('20')):
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