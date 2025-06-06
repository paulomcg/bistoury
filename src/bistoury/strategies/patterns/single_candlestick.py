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
        body_ratio = self._calculate_body_to_range_ratio(candle)
        upper_shadow_ratio = self._calculate_upper_shadow_ratio(candle)
        lower_shadow_ratio = self._calculate_lower_shadow_ratio(candle)
        
        # Doji requires very small body relative to range
        max_body_ratio = Decimal('0.05')  # 5% of range
        
        if body_ratio > max_body_ratio:
            return None
        
        # Calculate confidence based on how small the body is
        body_score = (max_body_ratio - body_ratio) / max_body_ratio * Decimal('100')
        
        # Bonus for symmetric shadows (more indecision)
        shadow_symmetry = Decimal('1') - abs(upper_shadow_ratio - lower_shadow_ratio)
        symmetry_bonus = shadow_symmetry * Decimal('20')
        
        confidence = min(Decimal('100'), body_score + symmetry_bonus)
        
        if confidence < self.min_confidence:
            return None
        
        # Determine Doji subtype
        doji_subtype = self._classify_doji_subtype(upper_shadow_ratio, lower_shadow_ratio)
        
        # Doji is neutral but can be bullish/bearish based on context
        # Standard weighting: slightly more neutral
        bullish_prob = Decimal('0.45')
        bearish_prob = Decimal('0.45')
        
        # Adjust probabilities based on subtype
        if doji_subtype == "dragonfly":
            bullish_prob = Decimal('0.65')
            bearish_prob = Decimal('0.35')
        elif doji_subtype == "gravestone":
            bullish_prob = Decimal('0.35')
            bearish_prob = Decimal('0.65')
        
        # Higher reliability for more perfect Doji
        reliability = Decimal('0.7') + (symmetry_bonus / Decimal('100'))
        
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
    
    def _classify_doji_subtype(self, upper_ratio: Decimal, lower_ratio: Decimal) -> str:
        """Classify the type of Doji based on shadow ratios."""
        min_long_shadow = Decimal('0.4')  # 40% of range
        max_short_shadow = Decimal('0.1')  # 10% of range
        
        if lower_ratio >= min_long_shadow and upper_ratio <= max_short_shadow:
            return "dragonfly"
        elif upper_ratio >= min_long_shadow and lower_ratio <= max_short_shadow:
            return "gravestone"
        elif upper_ratio >= min_long_shadow and lower_ratio >= min_long_shadow:
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
        body_ratio = self._calculate_body_to_range_ratio(candle)
        upper_shadow_ratio = self._calculate_upper_shadow_ratio(candle)
        lower_shadow_ratio = self._calculate_lower_shadow_ratio(candle)
        
        # Hammer requirements
        min_lower_shadow = Decimal('0.6')   # Lower shadow >= 60% of range
        max_upper_shadow = Decimal('0.15')  # Upper shadow <= 15% of range
        max_body_ratio = Decimal('0.3')     # Body <= 30% of range
        
        if (lower_shadow_ratio < min_lower_shadow or 
            upper_shadow_ratio > max_upper_shadow or 
            body_ratio > max_body_ratio):
            return None
        
        # Calculate confidence based on how well it fits criteria
        lower_shadow_score = min(Decimal('40'), (lower_shadow_ratio - min_lower_shadow) * Decimal('100'))
        upper_shadow_score = max(Decimal('0'), (max_upper_shadow - upper_shadow_ratio) * Decimal('200'))
        body_score = (max_body_ratio - body_ratio) / max_body_ratio * Decimal('30')
        
        confidence = Decimal('30') + lower_shadow_score + upper_shadow_score + body_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Hammer is typically bullish reversal
        # Bullish body (close > open) gives slight edge
        if candle.is_bullish:
            bullish_prob = Decimal('0.75')
            bearish_prob = Decimal('0.25')
        else:
            bullish_prob = Decimal('0.70')
            bearish_prob = Decimal('0.30')
        
        # Reliability based on how textbook the pattern is
        reliability = Decimal('0.75') + (confidence - Decimal('60')) / Decimal('200')
        reliability = min(Decimal('0.95'), max(Decimal('0.6'), reliability))
        
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
        
        # Shooting Star requirements (inverse of Hammer)
        min_upper_shadow = Decimal('0.6')   # Upper shadow >= 60% of range
        max_lower_shadow = Decimal('0.15')  # Lower shadow <= 15% of range
        max_body_ratio = Decimal('0.3')     # Body <= 30% of range
        
        if (upper_shadow_ratio < min_upper_shadow or 
            lower_shadow_ratio > max_lower_shadow or 
            body_ratio > max_body_ratio):
            return None
        
        # Calculate confidence (similar to Hammer but for upper shadow)
        upper_shadow_score = min(Decimal('40'), (upper_shadow_ratio - min_upper_shadow) * Decimal('100'))
        lower_shadow_score = max(Decimal('0'), (max_lower_shadow - lower_shadow_ratio) * Decimal('200'))
        body_score = (max_body_ratio - body_ratio) / max_body_ratio * Decimal('30')
        
        confidence = Decimal('30') + upper_shadow_score + lower_shadow_score + body_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Shooting Star is typically bearish reversal
        # Bearish body (close < open) gives slight edge
        if candle.is_bearish:
            bullish_prob = Decimal('0.25')
            bearish_prob = Decimal('0.75')
        else:
            bullish_prob = Decimal('0.30')
            bearish_prob = Decimal('0.70')
        
        # Reliability based on pattern quality
        reliability = Decimal('0.75') + (confidence - Decimal('60')) / Decimal('200')
        reliability = min(Decimal('0.95'), max(Decimal('0.6'), reliability))
        
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
        body_ratio = self._calculate_body_to_range_ratio(candle)
        upper_shadow_ratio = self._calculate_upper_shadow_ratio(candle)
        lower_shadow_ratio = self._calculate_lower_shadow_ratio(candle)
        
        # Spinning Top requirements
        max_body_ratio = Decimal('0.3')     # Small body
        min_shadow_ratio = Decimal('0.25')  # Decent shadows on both sides
        max_shadow_diff = Decimal('0.3')    # Shadows relatively similar
        
        shadow_difference = abs(upper_shadow_ratio - lower_shadow_ratio)
        
        if (body_ratio > max_body_ratio or 
            upper_shadow_ratio < min_shadow_ratio or 
            lower_shadow_ratio < min_shadow_ratio or
            shadow_difference > max_shadow_diff):
            return None
        
        # Calculate confidence based on shadow balance and body size
        shadow_balance_score = (max_shadow_diff - shadow_difference) / max_shadow_diff * Decimal('40')
        body_score = (max_body_ratio - body_ratio) / max_body_ratio * Decimal('30')
        shadow_length_score = min(Decimal('30'), (upper_shadow_ratio + lower_shadow_ratio) * Decimal('50'))
        
        confidence = shadow_balance_score + body_score + shadow_length_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Spinning Top is neutral with slight bias based on body color
        if candle.is_bullish:
            bullish_prob = Decimal('0.55')
            bearish_prob = Decimal('0.45')
        else:
            bullish_prob = Decimal('0.45')
            bearish_prob = Decimal('0.55')
        
        # Moderate reliability - pattern indicates indecision
        reliability = Decimal('0.65') + (confidence - Decimal('60')) / Decimal('250')
        reliability = min(Decimal('0.85'), max(Decimal('0.5'), reliability))
        
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
        body_ratio = self._calculate_body_to_range_ratio(candle)
        upper_shadow_ratio = self._calculate_upper_shadow_ratio(candle)
        lower_shadow_ratio = self._calculate_lower_shadow_ratio(candle)
        
        # Marubozu requirements
        min_body_ratio = Decimal('0.85')    # Large body (85%+ of range)
        max_shadow_ratio = Decimal('0.05')  # Minimal shadows (5% max)
        
        if (body_ratio < min_body_ratio or 
            upper_shadow_ratio > max_shadow_ratio or 
            lower_shadow_ratio > max_shadow_ratio):
            return None
        
        # Calculate confidence based on body size and lack of shadows
        body_score = (body_ratio - min_body_ratio) / (Decimal('1') - min_body_ratio) * Decimal('60')
        shadow_score = (max_shadow_ratio - max(upper_shadow_ratio, lower_shadow_ratio)) / max_shadow_ratio * Decimal('40')
        
        confidence = body_score + shadow_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Marubozu strongly indicates continuation of current direction
        if candle.is_bullish:
            bullish_prob = Decimal('0.85')
            bearish_prob = Decimal('0.15')
        else:
            bullish_prob = Decimal('0.15')
            bearish_prob = Decimal('0.85')
        
        # High reliability for strong directional patterns
        reliability = Decimal('0.8') + (confidence - Decimal('60')) / Decimal('200')
        reliability = min(Decimal('0.95'), max(Decimal('0.7'), reliability))
        
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