"""
Multi-Candlestick Pattern Recognition

This module implements recognition algorithms for multi-candlestick patterns
including Engulfing, Harami, Piercing Line, Dark Cloud Cover, Morning/Evening Star,
and Tweezer patterns.

Each pattern detector analyzes sequences of 2-3 candlesticks to identify:
- Pattern completion and validation
- Bullish/bearish probability based on context
- Pattern strength and reliability scores
- Volume confirmation across the pattern sequence

Built on top of existing CandlestickData and CandlestickPattern models.
"""

from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional, Dict, Any, List, Tuple
from abc import ABC, abstractmethod

from ...models.market_data import CandlestickData, Timeframe
from ...models.signals import CandlestickPattern, PatternType, SignalDirection
from ..candlestick_models import VolumeProfile
from .pattern_config import get_pattern_config
from .config_manager import initialize_pattern_config

# Initialize pattern configuration on module import
try:
    initialize_pattern_config()
except Exception as e:
    print(f"Warning: Could not initialize pattern configuration: {e}")
    print("Using default hardcoded values")

class MultiPatternDetector(ABC):
    """
    Abstract base class for multi-candlestick pattern detectors.
    
    Provides common functionality for analyzing sequences of candlesticks
    to identify complex patterns that require multiple candles.
    """
    
    def __init__(self, min_confidence: Decimal = Decimal('60')):
        self.min_confidence = min_confidence
    
    @abstractmethod
    def detect(self, candles: List[CandlestickData], volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Detect the specific multi-candlestick pattern."""
        pass
    
    @abstractmethod
    def get_pattern_type(self) -> PatternType:
        """Return the pattern type this detector identifies."""
        pass
    
    @abstractmethod
    def get_required_candles(self) -> int:
        """Return the number of candlesticks required for this pattern."""
        pass
    
    def _validate_candle_sequence(self, candles: List[CandlestickData]) -> bool:
        """Validate that candles form a proper sequence."""
        if len(candles) != self.get_required_candles():
            return False
        
        # Ensure candles are in chronological order
        for i in range(1, len(candles)):
            if candles[i].timestamp <= candles[i-1].timestamp:
                return False
        
        # Ensure all candles are from the same symbol and timeframe
        symbol = candles[0].symbol
        timeframe = candles[0].timeframe
        for candle in candles[1:]:
            if candle.symbol != symbol or candle.timeframe != timeframe:
                return False
        
        return True
    
    def _calculate_pattern_strength(self, candles: List[CandlestickData]) -> Decimal:
        """Calculate overall pattern strength based on candle characteristics."""
        if not candles:
            return Decimal('0')
        
        # Base strength on range and volume
        total_range = sum(candle.high - candle.low for candle in candles)
        avg_range = total_range / len(candles)
        
        # Strength increases with larger ranges (more decisive moves)
        range_score = min(Decimal('50'), avg_range / candles[0].close * Decimal('1000'))
        
        # Volume consistency adds to strength
        volumes = [candle.volume for candle in candles]
        avg_volume = sum(volumes) / len(volumes)
        volume_consistency = Decimal('50') - (max(volumes) - min(volumes)) / avg_volume * Decimal('25')
        volume_consistency = max(Decimal('0'), volume_consistency)
        
        return range_score + volume_consistency
    
    def _create_pattern(
        self,
        candles: List[CandlestickData],
        confidence: Decimal,
        reliability: Decimal,
        bullish_probability: Decimal,
        bearish_probability: Decimal,
        volume_profile: Optional[VolumeProfile] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> CandlestickPattern:
        """Create a CandlestickPattern instance for multi-candle patterns."""
        # Use the last candle for symbol/timeframe info and completion price
        last_candle = candles[-1]
        pattern_id = f"{self.get_pattern_type().value}_{last_candle.symbol}_{int(last_candle.timestamp.timestamp() * 1000)}"
        
        return CandlestickPattern(
            pattern_id=pattern_id,
            symbol=last_candle.symbol,
            pattern_type=self.get_pattern_type(),
            candles=candles,
            timeframe=last_candle.timeframe,
            confidence=confidence,
            reliability=reliability,
            bullish_probability=bullish_probability,
            bearish_probability=bearish_probability,
            completion_price=last_candle.close,
            volume_confirmation=volume_profile.volume_confirmation if volume_profile else False,
            pattern_metadata=metadata or {}
        )


class EngulfingDetector(MultiPatternDetector):
    """
    Engulfing pattern detector.
    
    An Engulfing pattern occurs when a larger candle completely engulfs
    the previous candle's body, indicating potential reversal:
    - Bullish Engulfing: Small bearish candle followed by large bullish candle
    - Bearish Engulfing: Small bullish candle followed by large bearish candle
    """
    
    def get_pattern_type(self) -> PatternType:
        return PatternType.ENGULFING
    
    def get_required_candles(self) -> int:
        return 2
    
    def detect(self, candles: List[CandlestickData], volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Detect Engulfing pattern."""
        if not self._validate_candle_sequence(candles):
            return None
        
        first_candle, second_candle = candles
        
        # Engulfing requirements:
        # 1. Second candle body completely engulfs first candle body
        # 2. Candles have opposite directions
        # 3. Second candle has significantly larger body
        
        first_body_top = max(first_candle.open, first_candle.close)
        first_body_bottom = min(first_candle.open, first_candle.close)
        second_body_top = max(second_candle.open, second_candle.close)
        second_body_bottom = min(second_candle.open, second_candle.close)
        
        # Check if second candle engulfs first candle's body
        body_engulfed = (second_body_top > first_body_top and 
                        second_body_bottom < first_body_bottom)
        
        if not body_engulfed:
            return None
        
        # Check opposite directions
        opposite_directions = (first_candle.is_bullish != second_candle.is_bullish)
        if not opposite_directions:
            return None
        
        # Calculate engulfing strength
        first_body_size = first_body_top - first_body_bottom
        second_body_size = second_body_top - second_body_bottom
        
        if first_body_size == 0:  # Avoid division by zero
            return None
        
        engulfing_ratio = second_body_size / first_body_size
        
        config = get_pattern_config().engulfing
        
        # Require meaningful engulfing
        if engulfing_ratio < config.min_engulfing_ratio:
            return None
        
        # Calculate confidence based on engulfing strength
        ratio_score = min(config.max_ratio_score, (engulfing_ratio - config.min_engulfing_ratio) * config.ratio_score_multiplier)
        volume_score = config.volume_confirmed_score if (volume_profile and volume_profile.volume_confirmation) else config.volume_unconfirmed_score
        pattern_strength = self._calculate_pattern_strength(candles)
        
        confidence = config.base_confidence + ratio_score + volume_score + (pattern_strength * config.pattern_strength_weight)
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Determine bullish or bearish engulfing
        is_bullish_engulfing = second_candle.is_bullish
        
        if is_bullish_engulfing:
            bullish_prob = config.bullish_bullish_prob
            bearish_prob = config.bullish_bearish_prob
            engulfing_type = "bullish"
        else:
            bullish_prob = config.bearish_bullish_prob
            bearish_prob = config.bearish_bearish_prob
            engulfing_type = "bearish"
        
        # Reliability increases with stronger engulfing
        reliability = config.base_reliability + min(config.max_reliability_bonus, (engulfing_ratio - Decimal('1')) * config.reliability_multiplier)
        
        metadata = {
            "engulfing_type": engulfing_type,
            "engulfing_ratio": float(engulfing_ratio),
            "first_candle_body_size": float(first_body_size),
            "second_candle_body_size": float(second_body_size),
            "volume_confirmation": volume_profile.volume_confirmation if volume_profile else False
        }
        
        return self._create_pattern(
            candles=candles,
            confidence=confidence,
            reliability=reliability,
            bullish_probability=bullish_prob,
            bearish_probability=bearish_prob,
            volume_profile=volume_profile,
            metadata=metadata
        )


class HaramiDetector(MultiPatternDetector):
    """
    Harami pattern detector.
    
    A Harami pattern occurs when a smaller candle is contained within
    the previous candle's body, indicating potential reversal:
    - The first candle has a large body
    - The second candle is completely contained within the first's body
    - Often indicates indecision after a strong move
    """
    
    def get_pattern_type(self) -> PatternType:
        return PatternType.HARAMI
    
    def get_required_candles(self) -> int:
        return 2
    
    def detect(self, candles: List[CandlestickData], volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Detect Harami pattern."""
        if not self._validate_candle_sequence(candles):
            return None
        
        first_candle, second_candle = candles
        
        # Harami requirements:
        # 1. First candle has a substantial body
        # 2. Second candle body is completely contained within first candle body
        # 3. Second candle is significantly smaller than first
        
        first_body_top = max(first_candle.open, first_candle.close)
        first_body_bottom = min(first_candle.open, first_candle.close)
        second_body_top = max(second_candle.open, second_candle.close)
        second_body_bottom = min(second_candle.open, second_candle.close)
        
        first_body_size = first_body_top - first_body_bottom
        second_body_size = second_body_top - second_body_bottom
        
        config = get_pattern_config().harami
        
        # Require substantial first candle body
        first_body_ratio = first_body_size / first_candle.close
        if first_body_ratio < config.min_first_body_ratio:
            return None
        
        # Check if second candle is contained within first candle's body
        body_contained = (second_body_top <= first_body_top and 
                         second_body_bottom >= first_body_bottom)
        
        if not body_contained:
            return None
        
        # Calculate containment ratio
        if first_body_size == 0:
            return None
        
        containment_ratio = second_body_size / first_body_size
        
        # Require meaningful containment (second candle should be much smaller)
        if containment_ratio > config.max_containment_ratio:
            return None
        
        # Calculate confidence based on containment quality
        containment_score = min(config.max_containment_score, (config.max_containment_ratio - containment_ratio) / config.max_containment_ratio * config.containment_score_multiplier)
        first_body_score = min(config.max_body_score, first_body_ratio * config.body_score_multiplier)
        volume_score = config.volume_confirmed_score if (volume_profile and volume_profile.volume_confirmation) else config.volume_unconfirmed_score
        
        confidence = config.base_confidence + containment_score + first_body_score + volume_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Harami indicates potential reversal but with less certainty than engulfing
        # Bias toward reversal of the first candle's direction
        if first_candle.is_bullish:
            # After bullish candle, expect bearish reversal
            bullish_prob = config.bearish_bullish_prob
            bearish_prob = config.bearish_bearish_prob
            harami_type = "bearish"
        else:
            # After bearish candle, expect bullish reversal  
            bullish_prob = config.bullish_bullish_prob
            bearish_prob = config.bullish_bearish_prob
            harami_type = "bullish"
        
        # Reliability moderate due to indecision nature
        reliability = config.base_reliability + min(config.max_reliability_bonus, (confidence - Decimal('60')) / Decimal('200'))
        
        metadata = {
            "harami_type": harami_type,
            "containment_ratio": float(containment_ratio),
            "first_body_ratio": float(first_body_ratio),
            "first_candle_direction": "bullish" if first_candle.is_bullish else "bearish",
            "second_candle_direction": "bullish" if second_candle.is_bullish else "bearish"
        }
        
        return self._create_pattern(
            candles=candles,
            confidence=confidence,
            reliability=reliability,
            bullish_probability=bullish_prob,
            bearish_probability=bearish_prob,
            volume_profile=volume_profile,
            metadata=metadata
        )


class PiercingLineDetector(MultiPatternDetector):
    """
    Piercing Line pattern detector.
    
    A Piercing Line is a bullish reversal pattern consisting of:
    - A bearish candle followed by a bullish candle
    - The bullish candle opens below the bearish candle's low
    - The bullish candle closes above the midpoint of the bearish candle's body
    """
    
    def get_pattern_type(self) -> PatternType:
        return PatternType.PIERCING_LINE
    
    def get_required_candles(self) -> int:
        return 2
    
    def detect(self, candles: List[CandlestickData], volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Detect Piercing Line pattern."""
        if not self._validate_candle_sequence(candles):
            return None
        
        first_candle, second_candle = candles
        
        # Piercing Line requirements:
        # 1. First candle is bearish with substantial body
        # 2. Second candle is bullish
        # 3. Second candle opens below first candle's low (gap down)
        # 4. Second candle closes above midpoint of first candle's body
        
        if not first_candle.is_bearish or not second_candle.is_bullish:
            return None
        
        first_body_top = first_candle.open  # Bearish: open > close
        first_body_bottom = first_candle.close
        first_body_size = first_body_top - first_body_bottom
        first_body_midpoint = (first_body_top + first_body_bottom) / Decimal('2')
        
        config = get_pattern_config().piercing_line
        
        # Require substantial first candle body
        first_body_ratio = first_body_size / first_candle.close
        if first_body_ratio < config.min_body_ratio:
            return None
        
        # Check gap down opening
        gap_down = second_candle.open < first_candle.low
        if not gap_down:
            return None
        
        # Check piercing (close above midpoint)
        piercing = second_candle.close > first_body_midpoint
        # Additional check for minimum pierce ratio
        pierce_distance = second_candle.close - first_body_midpoint
        pierce_ratio = pierce_distance / first_body_size
        if not piercing or pierce_ratio < config.min_pierce_ratio:
            return None
        
        # Calculate gap size
        gap_size = first_candle.low - second_candle.open
        gap_ratio = gap_size / first_candle.close
        
        # Calculate confidence
        pierce_score = min(config.max_pierce_score, pierce_ratio * config.pierce_score_multiplier)
        gap_score = min(config.max_gap_score, gap_ratio * config.gap_score_multiplier)
        body_score = min(config.max_body_score, first_body_ratio * config.body_score_multiplier)
        volume_score = config.volume_confirmed_score if (volume_profile and volume_profile.volume_confirmation) else config.volume_unconfirmed_score
        
        confidence = config.base_confidence + pierce_score + gap_score + body_score + volume_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Piercing Line is bullish reversal pattern
        bullish_prob = config.bullish_prob
        bearish_prob = config.bearish_prob
        
        # Reliability based on how deep the piercing goes
        reliability = config.base_reliability + min(config.max_reliability_bonus, pierce_ratio)
        
        metadata = {
            "gap_size": float(gap_size),
            "gap_ratio": float(gap_ratio),
            "pierce_distance": float(pierce_distance),
            "pierce_ratio": float(pierce_ratio),
            "first_body_midpoint": float(first_body_midpoint),
            "pattern_type": "piercing_line"
        }
        
        return self._create_pattern(
            candles=candles,
            confidence=confidence,
            reliability=reliability,
            bullish_probability=bullish_prob,
            bearish_probability=bearish_prob,
            volume_profile=volume_profile,
            metadata=metadata
        )


class DarkCloudCoverDetector(MultiPatternDetector):
    """
    Dark Cloud Cover pattern detector.
    
    A Dark Cloud Cover is a bearish reversal pattern consisting of:
    - A bullish candle followed by a bearish candle
    - The bearish candle opens above the bullish candle's high
    - The bearish candle closes below the midpoint of the bullish candle's body
    """
    
    def get_pattern_type(self) -> PatternType:
        return PatternType.DARK_CLOUD_COVER
    
    def get_required_candles(self) -> int:
        return 2
    
    def detect(self, candles: List[CandlestickData], volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Detect Dark Cloud Cover pattern."""
        if not self._validate_candle_sequence(candles):
            return None
        
        first_candle, second_candle = candles
        
        # Dark Cloud Cover requirements:
        # 1. First candle is bullish with substantial body
        # 2. Second candle is bearish
        # 3. Second candle opens above first candle's high (gap up)
        # 4. Second candle closes below midpoint of first candle's body
        
        if not first_candle.is_bullish or not second_candle.is_bearish:
            return None
        
        first_body_bottom = first_candle.open  # Bullish: close > open
        first_body_top = first_candle.close
        first_body_size = first_body_top - first_body_bottom
        first_body_midpoint = (first_body_top + first_body_bottom) / Decimal('2')
        
        config = get_pattern_config().dark_cloud_cover
        
        # Require substantial first candle body
        first_body_ratio = first_body_size / first_candle.close
        if first_body_ratio < config.min_body_ratio:
            return None
        
        # Check gap up opening
        gap_up = second_candle.open > first_candle.high
        if not gap_up:
            return None
        
        # Check cloud cover (close below midpoint)
        cloud_cover = second_candle.close < first_body_midpoint
        # Additional check for minimum cover ratio
        cover_distance = first_body_midpoint - second_candle.close
        cover_ratio = cover_distance / first_body_size
        if not cloud_cover or cover_ratio < config.min_cover_ratio:
            return None
        
        # Calculate gap size
        gap_size = second_candle.open - first_candle.high
        gap_ratio = gap_size / first_candle.close
        
        # Calculate confidence
        cover_score = min(config.max_cover_score, cover_ratio * config.cover_score_multiplier)
        gap_score = min(config.max_gap_score, gap_ratio * config.gap_score_multiplier)
        body_score = min(config.max_body_score, first_body_ratio * config.body_score_multiplier)
        volume_score = config.volume_confirmed_score if (volume_profile and volume_profile.volume_confirmation) else config.volume_unconfirmed_score
        
        confidence = config.base_confidence + cover_score + gap_score + body_score + volume_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Dark Cloud Cover is bearish reversal pattern
        bullish_prob = config.bullish_prob
        bearish_prob = config.bearish_prob
        
        # Reliability based on how deep the cloud cover goes
        reliability = config.base_reliability + min(config.max_reliability_bonus, cover_ratio)
        
        metadata = {
            "gap_size": float(gap_size),
            "gap_ratio": float(gap_ratio), 
            "cover_distance": float(cover_distance),
            "cover_ratio": float(cover_ratio),
            "first_body_midpoint": float(first_body_midpoint),
            "pattern_type": "dark_cloud_cover"
        }
        
        return self._create_pattern(
            candles=candles,
            confidence=confidence,
            reliability=reliability,
            bullish_probability=bullish_prob,
            bearish_probability=bearish_prob,
            volume_profile=volume_profile,
            metadata=metadata
        )


class MorningStarDetector(MultiPatternDetector):
    """
    Morning Star pattern detector.
    
    A Morning Star is a bullish reversal pattern consisting of three candles:
    1. A bearish candle with substantial body
    2. A small-bodied candle (star) that gaps down
    3. A bullish candle that closes above the midpoint of the first candle
    """
    
    def get_pattern_type(self) -> PatternType:
        return PatternType.MORNING_STAR
    
    def get_required_candles(self) -> int:
        return 3
    
    def detect(self, candles: List[CandlestickData], volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Detect Morning Star pattern."""
        if not self._validate_candle_sequence(candles):
            return None
        
        first_candle, star_candle, third_candle = candles
        
        # Morning Star requirements:
        # 1. First candle is bearish with substantial body
        # 2. Star candle has small body and gaps down from first candle
        # 3. Third candle is bullish and closes above midpoint of first candle
        
        if not first_candle.is_bearish or not third_candle.is_bullish:
            return None
        
        # Calculate body sizes
        first_body_size = first_candle.open - first_candle.close  # Bearish
        star_body_size = abs(star_candle.close - star_candle.open)
        third_body_size = third_candle.close - third_candle.open  # Bullish
        
        config = get_pattern_config().morning_star
        
        # Require substantial first candle body
        first_body_ratio = first_body_size / first_candle.close
        if first_body_ratio < config.min_first_body_ratio:
            return None
        
        # Require small star body relative to first candle
        star_ratio = star_body_size / first_body_size if first_body_size > 0 else Decimal('1')
        if star_ratio > config.max_star_ratio:
            return None
        
        # Check gaps
        star_top = max(star_candle.open, star_candle.close)
        star_bottom = min(star_candle.open, star_candle.close)
        
        # Gap down between first and star
        gap_down = star_top < first_candle.close
        if not gap_down:
            return None
        
        # Third candle should close above midpoint of first candle
        first_midpoint = (first_candle.open + first_candle.close) / Decimal('2')
        recovery = third_candle.close > first_midpoint
        if not recovery:
            return None
        
        # Calculate pattern strength
        gap_size = first_candle.close - star_top
        gap_ratio = gap_size / first_candle.close
        recovery_distance = third_candle.close - first_midpoint
        recovery_ratio = recovery_distance / first_body_size
        
        # Calculate confidence
        gap_score = min(config.max_gap_score, gap_ratio * config.gap_score_multiplier)
        star_score = (config.max_star_ratio - star_ratio) / config.max_star_ratio * config.star_score_weight
        recovery_score = min(config.max_recovery_score, recovery_ratio * config.recovery_score_multiplier)
        volume_score = config.volume_confirmed_score if (volume_profile and volume_profile.volume_confirmation) else config.volume_unconfirmed_score
        
        confidence = gap_score + star_score + recovery_score + volume_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Morning Star is bullish reversal pattern
        bullish_prob = config.bullish_prob
        bearish_prob = config.bearish_prob
        
        # High reliability for three-candle reversal patterns
        reliability = config.base_reliability + min(config.max_reliability_bonus, recovery_ratio)
        
        metadata = {
            "gap_size": float(gap_size),
            "gap_ratio": float(gap_ratio),
            "star_body_ratio": float(star_ratio),
            "recovery_distance": float(recovery_distance),
            "recovery_ratio": float(recovery_ratio),
            "star_direction": "bullish" if star_candle.is_bullish else "bearish",
            "pattern_type": "morning_star"
        }
        
        return self._create_pattern(
            candles=candles,
            confidence=confidence,
            reliability=reliability,
            bullish_probability=bullish_prob,
            bearish_probability=bearish_prob,
            volume_profile=volume_profile,
            metadata=metadata
        )


class EveningStarDetector(MultiPatternDetector):
    """
    Evening Star pattern detector.
    
    An Evening Star is a bearish reversal pattern consisting of three candles:
    1. A bullish candle with substantial body
    2. A small-bodied candle (star) that gaps up
    3. A bearish candle that closes below the midpoint of the first candle
    """
    
    def get_pattern_type(self) -> PatternType:
        return PatternType.EVENING_STAR
    
    def get_required_candles(self) -> int:
        return 3
    
    def detect(self, candles: List[CandlestickData], volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Detect Evening Star pattern."""
        if not self._validate_candle_sequence(candles):
            return None
        
        first_candle, star_candle, third_candle = candles
        
        # Evening Star requirements:
        # 1. First candle is bullish with substantial body
        # 2. Star candle has small body and gaps up from first candle
        # 3. Third candle is bearish and closes below midpoint of first candle
        
        if not first_candle.is_bullish or not third_candle.is_bearish:
            return None
        
        # Calculate body sizes
        first_body_size = first_candle.close - first_candle.open  # Bullish
        star_body_size = abs(star_candle.close - star_candle.open)
        third_body_size = third_candle.open - third_candle.close  # Bearish
        
        config = get_pattern_config().evening_star
        
        # Require substantial first candle body
        first_body_ratio = first_body_size / first_candle.close
        if first_body_ratio < config.min_first_body_ratio:
            return None
        
        # Require small star body relative to first candle
        star_ratio = star_body_size / first_body_size if first_body_size > 0 else Decimal('1')
        if star_ratio > config.max_star_ratio:
            return None
        
        # Check gaps
        star_top = max(star_candle.open, star_candle.close)
        star_bottom = min(star_candle.open, star_candle.close)
        
        # Gap up between first and star
        gap_up = star_bottom > first_candle.close
        if not gap_up:
            return None
        
        # Third candle should close below midpoint of first candle
        first_midpoint = (first_candle.open + first_candle.close) / Decimal('2')
        decline = third_candle.close < first_midpoint
        if not decline:
            return None
        
        # Calculate pattern strength
        gap_size = star_bottom - first_candle.close
        gap_ratio = gap_size / first_candle.close
        decline_distance = first_midpoint - third_candle.close
        decline_ratio = decline_distance / first_body_size
        
        # Calculate confidence
        gap_score = min(config.max_gap_score, gap_ratio * config.gap_score_multiplier)
        star_score = (config.max_star_ratio - star_ratio) / config.max_star_ratio * config.star_score_weight
        decline_score = min(config.max_decline_score, decline_ratio * config.decline_score_multiplier)
        volume_score = config.volume_confirmed_score if (volume_profile and volume_profile.volume_confirmation) else config.volume_unconfirmed_score
        
        confidence = gap_score + star_score + decline_score + volume_score
        confidence = min(Decimal('100'), confidence)
        
        if confidence < self.min_confidence:
            return None
        
        # Evening Star is bearish reversal pattern
        bullish_prob = config.bullish_prob
        bearish_prob = config.bearish_prob
        
        # High reliability for three-candle reversal patterns
        reliability = config.base_reliability + min(config.max_reliability_bonus, decline_ratio)
        
        metadata = {
            "gap_size": float(gap_size),
            "gap_ratio": float(gap_ratio),
            "star_body_ratio": float(star_ratio),
            "decline_distance": float(decline_distance),
            "decline_ratio": float(decline_ratio),
            "star_direction": "bullish" if star_candle.is_bullish else "bearish",
            "pattern_type": "evening_star"
        }
        
        return self._create_pattern(
            candles=candles,
            confidence=confidence,
            reliability=reliability,
            bullish_probability=bullish_prob,
            bearish_probability=bearish_prob,
            volume_profile=volume_profile,
            metadata=metadata
        )


class MultiPatternRecognizer:
    """
    Comprehensive multi-candlestick pattern recognizer.
    
    Analyzes sequences of candlesticks to identify multiple pattern types
    and provides comprehensive pattern analysis and ranking.
    """
    
    def __init__(self, min_confidence: Decimal = Decimal('60')):
        self.min_confidence = min_confidence
        
        # Initialize all pattern detectors
        self.two_candle_detectors = [
            EngulfingDetector(min_confidence),
            HaramiDetector(min_confidence),
            PiercingLineDetector(min_confidence),
            DarkCloudCoverDetector(min_confidence)
        ]
        
        self.three_candle_detectors = [
            MorningStarDetector(min_confidence),
            EveningStarDetector(min_confidence)
        ]
        
        self.all_detectors = self.two_candle_detectors + self.three_candle_detectors
    
    def analyze(self, candles: List[CandlestickData], volume_profile: Optional[VolumeProfile] = None) -> List[CandlestickPattern]:
        """
        Analyze candlestick sequence for all multi-candlestick patterns.
        
        Args:
            candles: List of candlesticks in chronological order (minimum 2, maximum depends on patterns)
            volume_profile: Optional volume analysis for pattern confirmation
            
        Returns:
            List of detected patterns sorted by confidence (highest first)
        """
        if len(candles) < 2:
            return []
        
        patterns = []
        
        # Check two-candle patterns if we have at least 2 candles
        if len(candles) >= 2:
            last_two = candles[-2:]
            for detector in self.two_candle_detectors:
                pattern = detector.detect(last_two, volume_profile)
                if pattern:
                    patterns.append(pattern)
        
        # Check three-candle patterns if we have at least 3 candles
        if len(candles) >= 3:
            last_three = candles[-3:]
            for detector in self.three_candle_detectors:
                pattern = detector.detect(last_three, volume_profile)
                if pattern:
                    patterns.append(pattern)
        
        # Sort by confidence (highest first)
        patterns.sort(key=lambda p: p.confidence, reverse=True)
        return patterns
    
    def get_best_pattern(self, candles: List[CandlestickData], volume_profile: Optional[VolumeProfile] = None) -> Optional[CandlestickPattern]:
        """Get the highest confidence pattern from the analysis."""
        patterns = self.analyze(candles, volume_profile)
        return patterns[0] if patterns else None
    
    def get_pattern_summary(self, candles: List[CandlestickData], volume_profile: Optional[VolumeProfile] = None) -> Dict[str, Any]:
        """
        Get comprehensive summary of all detected patterns.
        
        Returns:
            Dictionary with pattern statistics and analysis
        """
        patterns = self.analyze(candles, volume_profile)
        
        if not patterns:
            return {
                "total_patterns": 0,
                "best_pattern": None,
                "avg_confidence": Decimal('0'),
                "bullish_patterns": 0,
                "bearish_patterns": 0,
                "reversal_patterns": 0,
                "pattern_types": []
            }
        
        # Calculate statistics
        total_confidence = sum(p.confidence for p in patterns)
        avg_confidence = total_confidence / len(patterns)
        
        bullish_patterns = [p for p in patterns if p.bullish_probability > p.bearish_probability]
        bearish_patterns = [p for p in patterns if p.bearish_probability > p.bullish_probability]
        reversal_patterns = [p for p in patterns if p.is_reversal_pattern]
        
        return {
            "total_patterns": len(patterns),
            "best_pattern": patterns[0].pattern_type.value,
            "best_confidence": patterns[0].confidence,
            "avg_confidence": avg_confidence,
            "bullish_patterns": len(bullish_patterns),
            "bearish_patterns": len(bearish_patterns),
            "reversal_patterns": len(reversal_patterns),
            "pattern_types": [p.pattern_type.value for p in patterns]
        } 