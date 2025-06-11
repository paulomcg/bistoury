"""
Pattern Detection Configuration

This module defines all configurable parameters for candlestick pattern detection.
All hardcoded thresholds and ratios are centralized here for easy tweaking.
"""

from decimal import Decimal
from dataclasses import dataclass, field
from typing import Dict, Any
import json
from pathlib import Path


@dataclass
class DojiConfig:
    """Configuration for Doji pattern detection."""
    max_body_ratio: Decimal = Decimal('0.02')  # 2% of range
    min_range_ratio: Decimal = Decimal('0.005')  # 0.5% of price
    body_score_weight: Decimal = Decimal('60')
    symmetry_bonus_weight: Decimal = Decimal('20')
    range_bonus_weight: Decimal = Decimal('20')
    range_bonus_multiplier: Decimal = Decimal('2000')
    
    # Doji subtype classification
    min_long_shadow: Decimal = Decimal('0.4')  # 40% of range
    max_short_shadow: Decimal = Decimal('0.1')  # 10% of range
    
    # Probability adjustments
    dragonfly_bullish_prob: Decimal = Decimal('0.65')
    dragonfly_bearish_prob: Decimal = Decimal('0.35')
    gravestone_bullish_prob: Decimal = Decimal('0.35')
    gravestone_bearish_prob: Decimal = Decimal('0.65')
    standard_bullish_prob: Decimal = Decimal('0.45')
    standard_bearish_prob: Decimal = Decimal('0.45')
    
    # Reliability
    base_reliability: Decimal = Decimal('0.7')


@dataclass
class HammerConfig:
    """Configuration for Hammer pattern detection."""
    min_lower_shadow: Decimal = Decimal('0.6')   # 60% of range
    max_upper_shadow: Decimal = Decimal('0.15')  # 15% of range
    max_body_ratio: Decimal = Decimal('0.3')     # 30% of range
    
    # Confidence scoring
    base_confidence: Decimal = Decimal('30')
    lower_shadow_score_multiplier: Decimal = Decimal('100')
    upper_shadow_score_multiplier: Decimal = Decimal('200')
    body_score_weight: Decimal = Decimal('30')
    max_lower_shadow_score: Decimal = Decimal('40')
    
    # Probabilities
    bullish_body_bullish_prob: Decimal = Decimal('0.75')
    bullish_body_bearish_prob: Decimal = Decimal('0.25')
    bearish_body_bullish_prob: Decimal = Decimal('0.70')
    bearish_body_bearish_prob: Decimal = Decimal('0.30')
    
    # Reliability
    base_reliability: Decimal = Decimal('0.75')
    reliability_divisor: Decimal = Decimal('200')
    max_reliability: Decimal = Decimal('0.95')
    min_reliability: Decimal = Decimal('0.6')


@dataclass
class ShootingStarConfig:
    """Configuration for Shooting Star pattern detection."""
    min_upper_shadow: Decimal = Decimal('0.6')   # 60% of range
    max_lower_shadow: Decimal = Decimal('0.15')  # 15% of range
    max_body_ratio: Decimal = Decimal('0.3')     # 30% of range
    
    # Confidence scoring
    base_confidence: Decimal = Decimal('30')
    upper_shadow_score_multiplier: Decimal = Decimal('100')
    lower_shadow_score_multiplier: Decimal = Decimal('200')
    body_score_weight: Decimal = Decimal('30')
    max_upper_shadow_score: Decimal = Decimal('40')
    
    # Probabilities
    bearish_body_bullish_prob: Decimal = Decimal('0.25')
    bearish_body_bearish_prob: Decimal = Decimal('0.75')
    bullish_body_bullish_prob: Decimal = Decimal('0.30')
    bullish_body_bearish_prob: Decimal = Decimal('0.70')
    
    # Reliability
    base_reliability: Decimal = Decimal('0.75')
    reliability_divisor: Decimal = Decimal('200')
    max_reliability: Decimal = Decimal('0.95')
    min_reliability: Decimal = Decimal('0.6')


@dataclass
class SpinningTopConfig:
    """Configuration for Spinning Top pattern detection."""
    max_body_ratio: Decimal = Decimal('0.3')     # 30% body max
    min_shadow_ratio: Decimal = Decimal('0.25')  # 25% shadows min
    max_shadow_diff: Decimal = Decimal('0.3')    # 30% max difference
    
    # Confidence scoring
    shadow_balance_weight: Decimal = Decimal('40')
    body_score_weight: Decimal = Decimal('30')
    shadow_length_weight: Decimal = Decimal('30')
    shadow_length_multiplier: Decimal = Decimal('50')
    
    # Probabilities
    bullish_body_bullish_prob: Decimal = Decimal('0.55')
    bullish_body_bearish_prob: Decimal = Decimal('0.45')
    bearish_body_bullish_prob: Decimal = Decimal('0.45')
    bearish_body_bearish_prob: Decimal = Decimal('0.55')
    
    # Reliability
    base_reliability: Decimal = Decimal('0.65')
    reliability_divisor: Decimal = Decimal('250')
    max_reliability: Decimal = Decimal('0.85')
    min_reliability: Decimal = Decimal('0.5')


@dataclass
class MarubozuConfig:
    """Configuration for Marubozu pattern detection."""
    min_body_ratio: Decimal = Decimal('0.85')    # 85% of range
    max_shadow_ratio: Decimal = Decimal('0.05')  # 5% max shadows
    
    # Confidence scoring
    body_score_weight: Decimal = Decimal('60')
    shadow_score_weight: Decimal = Decimal('40')
    
    # Probabilities
    bullish_bullish_prob: Decimal = Decimal('0.85')
    bullish_bearish_prob: Decimal = Decimal('0.15')
    bearish_bullish_prob: Decimal = Decimal('0.15')
    bearish_bearish_prob: Decimal = Decimal('0.85')
    
    # Reliability
    base_reliability: Decimal = Decimal('0.8')
    reliability_divisor: Decimal = Decimal('200')
    max_reliability: Decimal = Decimal('0.95')
    min_reliability: Decimal = Decimal('0.7')


@dataclass
class EngulfingConfig:
    """Configuration for Engulfing pattern detection."""
    min_engulfing_ratio: Decimal = Decimal('1.5')  # 1.5x larger body
    
    # Confidence scoring
    base_confidence: Decimal = Decimal('30')
    ratio_score_multiplier: Decimal = Decimal('25')
    max_ratio_score: Decimal = Decimal('50')
    volume_confirmed_score: Decimal = Decimal('20')
    volume_unconfirmed_score: Decimal = Decimal('10')
    pattern_strength_weight: Decimal = Decimal('0.4')
    
    # Probabilities
    bullish_bullish_prob: Decimal = Decimal('0.75')
    bullish_bearish_prob: Decimal = Decimal('0.25')
    bearish_bullish_prob: Decimal = Decimal('0.25')
    bearish_bearish_prob: Decimal = Decimal('0.75')
    
    # Reliability
    base_reliability: Decimal = Decimal('0.7')
    reliability_multiplier: Decimal = Decimal('0.1')
    max_reliability_bonus: Decimal = Decimal('0.2')


@dataclass
class HaramiConfig:
    """Configuration for Harami pattern detection."""
    min_first_body_ratio: Decimal = Decimal('0.01')  # 1% body ratio (reduced from 2%)
    max_containment_ratio: Decimal = Decimal('0.6')  # 60% containment max
    
    # Confidence scoring
    base_confidence: Decimal = Decimal('20')
    containment_score_multiplier: Decimal = Decimal('100')
    max_containment_score: Decimal = Decimal('40')
    body_score_multiplier: Decimal = Decimal('1000')
    max_body_score: Decimal = Decimal('25')
    volume_confirmed_score: Decimal = Decimal('20')
    volume_unconfirmed_score: Decimal = Decimal('5')
    
    # Probabilities
    bullish_bullish_prob: Decimal = Decimal('0.70')
    bullish_bearish_prob: Decimal = Decimal('0.30')
    bearish_bullish_prob: Decimal = Decimal('0.30')
    bearish_bearish_prob: Decimal = Decimal('0.70')
    
    # Reliability
    base_reliability: Decimal = Decimal('0.65')
    max_reliability_bonus: Decimal = Decimal('0.25')


@dataclass
class PiercingLineConfig:
    """Configuration for Piercing Line pattern detection."""
    min_body_ratio: Decimal = Decimal('0.008')  # 0.8% body ratio (reduced from 1.5%)
    min_pierce_ratio: Decimal = Decimal('0.5')  # 50% penetration
    
    # Confidence scoring
    base_confidence: Decimal = Decimal('10')
    pierce_score_multiplier: Decimal = Decimal('100')
    max_pierce_score: Decimal = Decimal('40')
    gap_score_multiplier: Decimal = Decimal('500')
    max_gap_score: Decimal = Decimal('25')
    body_score_multiplier: Decimal = Decimal('1000')
    max_body_score: Decimal = Decimal('25')
    volume_confirmed_score: Decimal = Decimal('20')
    volume_unconfirmed_score: Decimal = Decimal('5')
    
    # Probabilities
    bullish_prob: Decimal = Decimal('0.75')
    bearish_prob: Decimal = Decimal('0.25')
    
    # Reliability
    base_reliability: Decimal = Decimal('0.65')
    max_reliability_bonus: Decimal = Decimal('0.25')


@dataclass
class DarkCloudCoverConfig:
    """Configuration for Dark Cloud Cover pattern detection."""
    min_body_ratio: Decimal = Decimal('0.008')  # 0.8% body ratio (reduced from 1.5%)
    min_cover_ratio: Decimal = Decimal('0.5')   # 50% penetration
    
    # Confidence scoring
    base_confidence: Decimal = Decimal('10')
    cover_score_multiplier: Decimal = Decimal('100')
    max_cover_score: Decimal = Decimal('40')
    gap_score_multiplier: Decimal = Decimal('500')
    max_gap_score: Decimal = Decimal('25')
    body_score_multiplier: Decimal = Decimal('1000')
    max_body_score: Decimal = Decimal('25')
    volume_confirmed_score: Decimal = Decimal('20')
    volume_unconfirmed_score: Decimal = Decimal('5')
    
    # Probabilities
    bullish_prob: Decimal = Decimal('0.25')
    bearish_prob: Decimal = Decimal('0.75')
    
    # Reliability
    base_reliability: Decimal = Decimal('0.65')
    max_reliability_bonus: Decimal = Decimal('0.25')


@dataclass
class MorningStarConfig:
    """Configuration for Morning Star pattern detection."""
    min_first_body_ratio: Decimal = Decimal('0.008')  # 0.8% body ratio (reduced from 1.5%)
    max_star_ratio: Decimal = Decimal('0.3')         # 30% star body max
    
    # Confidence scoring
    gap_score_multiplier: Decimal = Decimal('1000')
    max_gap_score: Decimal = Decimal('25')
    star_score_weight: Decimal = Decimal('25')
    recovery_score_multiplier: Decimal = Decimal('100')
    max_recovery_score: Decimal = Decimal('30')
    volume_confirmed_score: Decimal = Decimal('20')
    volume_unconfirmed_score: Decimal = Decimal('10')
    
    # Probabilities
    bullish_prob: Decimal = Decimal('0.8')
    bearish_prob: Decimal = Decimal('0.2')
    
    # Reliability
    base_reliability: Decimal = Decimal('0.75')
    max_reliability_bonus: Decimal = Decimal('0.2')


@dataclass
class EveningStarConfig:
    """Configuration for Evening Star pattern detection."""
    min_first_body_ratio: Decimal = Decimal('0.008')  # 0.8% body ratio (reduced from 1.5%)
    max_star_ratio: Decimal = Decimal('0.3')         # 30% star body max
    
    # Confidence scoring
    gap_score_multiplier: Decimal = Decimal('1000')
    max_gap_score: Decimal = Decimal('25')
    star_score_weight: Decimal = Decimal('25')
    decline_score_multiplier: Decimal = Decimal('100')
    max_decline_score: Decimal = Decimal('30')
    volume_confirmed_score: Decimal = Decimal('20')
    volume_unconfirmed_score: Decimal = Decimal('10')
    
    # Probabilities
    bullish_prob: Decimal = Decimal('0.2')
    bearish_prob: Decimal = Decimal('0.8')
    
    # Reliability
    base_reliability: Decimal = Decimal('0.75')
    max_reliability_bonus: Decimal = Decimal('0.2')


@dataclass
class PatternDetectionConfig:
    """Master configuration for all pattern detection parameters."""
    
    # Global settings
    default_min_confidence: Decimal = Decimal('60')
    min_confidence_threshold: Decimal = Decimal('30')  # Signal generator threshold
    
    # Pattern-specific configurations
    doji: DojiConfig = field(default_factory=DojiConfig)
    hammer: HammerConfig = field(default_factory=HammerConfig)
    shooting_star: ShootingStarConfig = field(default_factory=ShootingStarConfig)
    spinning_top: SpinningTopConfig = field(default_factory=SpinningTopConfig)
    marubozu: MarubozuConfig = field(default_factory=MarubozuConfig)
    
    engulfing: EngulfingConfig = field(default_factory=EngulfingConfig)
    harami: HaramiConfig = field(default_factory=HaramiConfig)
    piercing_line: PiercingLineConfig = field(default_factory=PiercingLineConfig)
    dark_cloud_cover: DarkCloudCoverConfig = field(default_factory=DarkCloudCoverConfig)
    morning_star: MorningStarConfig = field(default_factory=MorningStarConfig)
    evening_star: EveningStarConfig = field(default_factory=EveningStarConfig)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for JSON serialization."""
        def convert_decimal(obj):
            if isinstance(obj, Decimal):
                return str(obj)
            elif isinstance(obj, dict):
                return {k: convert_decimal(v) for k, v in obj.items()}
            elif hasattr(obj, '__dict__'):
                return convert_decimal(obj.__dict__)
            return obj
        
        return convert_decimal(self.__dict__)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PatternDetectionConfig':
        """Create configuration from dictionary (JSON deserialization)."""
        def convert_to_decimal(obj):
            if isinstance(obj, str):
                try:
                    return Decimal(obj)
                except:
                    return obj
            elif isinstance(obj, dict):
                return {k: convert_to_decimal(v) for k, v in obj.items()}
            return obj
        
        # Convert string decimals back to Decimal objects
        converted_data = convert_to_decimal(data)
        
        # Create instances of sub-configs
        config_classes = {
            'doji': DojiConfig,
            'hammer': HammerConfig,
            'shooting_star': ShootingStarConfig,
            'spinning_top': SpinningTopConfig,
            'marubozu': MarubozuConfig,
            'engulfing': EngulfingConfig,
            'harami': HaramiConfig,
            'piercing_line': PiercingLineConfig,
            'dark_cloud_cover': DarkCloudCoverConfig,
            'morning_star': MorningStarConfig,
            'evening_star': EveningStarConfig
        }
        
        # Build the main config
        main_config_data = {}
        for key, value in converted_data.items():
            if key in config_classes:
                main_config_data[key] = config_classes[key](**value)
            else:
                main_config_data[key] = value
        
        return cls(**main_config_data)
    
    def save_to_file(self, filepath: Path):
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def load_from_file(cls, filepath: Path) -> 'PatternDetectionConfig':
        """Load configuration from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        return cls.from_dict(data)


# Global configuration instance
_config: PatternDetectionConfig = None


def get_pattern_config() -> PatternDetectionConfig:
    """Get the global pattern detection configuration."""
    global _config
    if _config is None:
        # Try to load from centralized config first, then fallback to defaults
        try:
            from ...config_manager import get_config_manager
            config_manager = get_config_manager()
            config_data = config_manager.get_section('pattern_detection')
            if config_data:
                _config = PatternDetectionConfig.from_dict(config_data)
            else:
                _config = PatternDetectionConfig()  # Use defaults
        except Exception:
            _config = PatternDetectionConfig()  # Fallback to defaults
    return _config


def set_pattern_config(config: PatternDetectionConfig):
    """Set the global pattern detection configuration."""
    global _config
    _config = config


def load_pattern_config(filepath: Path):
    """Load pattern configuration from file and set it as global."""
    config = PatternDetectionConfig.load_from_file(filepath)
    set_pattern_config(config)


def save_pattern_config(filepath: Path):
    """Save current global configuration to file."""
    config = get_pattern_config()
    config.save_to_file(filepath)


def reset_pattern_config():
    """Reset to default configuration."""
    global _config
    _config = PatternDetectionConfig() 