# Pattern Detection Configuration System

This system provides a centralized configuration for all candlestick pattern detection parameters, making it easy to tweak and optimize pattern recognition without modifying code.

## Overview

All pattern detection thresholds, ratios, and parameters are now configurable through a JSON file. This includes:

- **Single Pattern Detection**: Doji, Hammer, Shooting Star, Spinning Top, Marubozu
- **Multi-Pattern Detection**: Engulfing, Harami, Piercing Line, Dark Cloud Cover, Morning Star, Evening Star
- **Global Settings**: Default confidence thresholds, reliability calculations

## Quick Start

### 1. Create Your Configuration File

```bash
cd /path/to/your/project
python -m bistoury.strategies.patterns.config_manager create-config
```

This creates a `pattern_config.json` file in your project root with all default values.

### 2. Edit Configuration Parameters

Open `pattern_config.json` and modify values as needed:

```json
{
  "default_min_confidence": "60",
  "doji": {
    "max_body_ratio": "0.02",
    "min_range_ratio": "0.005",
    "dragonfly_bullish_prob": "0.65"
  },
  "harami": {
    "min_first_body_ratio": "0.01",
    "max_containment_ratio": "0.6"
  }
}
```

### 3. Configuration is Automatically Loaded

The pattern detection system automatically loads your configuration when imported. Changes take effect immediately on the next import or when you reload.

## Configuration Parameters

### Global Settings

- `default_min_confidence`: Default minimum confidence threshold (0-100)
- `min_confidence_threshold`: Signal generator minimum threshold

### Pattern-Specific Settings

#### Doji Pattern
- `max_body_ratio`: Maximum body size as ratio of total range (e.g., "0.02" = 2%)
- `min_range_ratio`: Minimum meaningful range as ratio of price
- `dragonfly_bullish_prob`/`gravestone_bearish_prob`: Probability adjustments for subtypes

#### Hammer/Shooting Star Patterns
- `min_lower_shadow`/`min_upper_shadow`: Required shadow size ratios
- `max_upper_shadow`/`max_lower_shadow`: Maximum opposite shadow ratios
- `max_body_ratio`: Maximum body size ratio

#### Multi-Candle Patterns
- `min_engulfing_ratio`: How much larger the engulfing candle must be (e.g., "1.5" = 1.5x)
- `min_body_ratio`: Minimum body size requirements
- `min_pierce_ratio`/`min_cover_ratio`: Penetration requirements for reversal patterns

## Management Commands

### View Current Configuration
```bash
python -m bistoury.strategies.patterns.config_manager info
```

### Validate Configuration
```bash
python -m bistoury.strategies.patterns.config_manager validate
```

### Backup Current Configuration
```bash
python -m bistoury.strategies.patterns.config_manager backup
```

### Reset to Defaults
```bash
python -m bistoury.strategies.patterns.config_manager reset
```

### Reload Configuration (if changed externally)
```bash
python -m bistoury.strategies.patterns.config_manager reload
```

## Configuration Examples

### More Sensitive Pattern Detection

Reduce thresholds to catch more patterns (but potentially more false positives):

```json
{
  "default_min_confidence": "40",
  "doji": {
    "max_body_ratio": "0.03"
  },
  "harami": {
    "min_first_body_ratio": "0.005",
    "max_containment_ratio": "0.7"
  },
  "engulfing": {
    "min_engulfing_ratio": "1.2"
  }
}
```

### More Conservative Detection

Increase thresholds for higher confidence patterns:

```json
{
  "default_min_confidence": "70",
  "doji": {
    "max_body_ratio": "0.015"
  },
  "engulfing": {
    "min_engulfing_ratio": "2.0"
  },
  "piercing_line": {
    "min_pierce_ratio": "0.6"
  }
}
```

### Custom Probability Weighting

Adjust bullish/bearish probabilities based on your market bias:

```json
{
  "hammer": {
    "bullish_body_bullish_prob": "0.85",
    "bearish_body_bullish_prob": "0.80"
  },
  "morning_star": {
    "bullish_prob": "0.85",
    "bearish_prob": "0.15"
  }
}
```

## Pattern Parameter Reference

### Body Ratio Parameters
- **Purpose**: Control how large/small a candle body must be
- **Format**: Decimal ratio (e.g., "0.02" = 2% of candle range)
- **Lower values**: More restrictive, fewer patterns detected
- **Higher values**: More permissive, more patterns detected

### Confidence Scoring
- **Purpose**: Weight different aspects of pattern quality
- **Format**: Decimal weights and multipliers
- **Higher weights**: More influence on final confidence score

### Probability Settings
- **Purpose**: Set bullish/bearish bias for each pattern
- **Format**: Decimal probabilities (must sum to ≤ 1.0)
- **Example**: "bullish_prob": "0.75" means 75% bullish probability

### Volume Scoring
- **Purpose**: Bonus points for volume confirmation
- **Format**: Point values added to confidence
- **Higher values**: More emphasis on volume confirmation

## Troubleshooting

### Configuration Not Loading
1. Check that `pattern_config.json` exists in your project root
2. Validate JSON syntax with: `python -m bistoury.strategies.patterns.config_manager validate`
3. Check console for error messages during import

### Invalid Values
- All ratio parameters should be decimal strings (e.g., "0.02", not 0.02)
- Probability values should be between "0.0" and "1.0"
- Confidence values should be between "0" and "100"

### Restoring Defaults
If your configuration gets corrupted, restore defaults:
```bash
python -m bistoury.strategies.patterns.config_manager reset
```

## File Locations

- **Default Configuration**: `src/bistoury/strategies/patterns/default_pattern_config.json`
- **User Configuration**: `pattern_config.json` (in your project root)
- **Backups**: `pattern_config.json.backup`, `pattern_config.json.backup.1`, etc.

## Best Practices

1. **Always backup** before making major changes
2. **Test incrementally** - change one parameter at a time
3. **Validate after changes** to catch syntax errors
4. **Document your changes** - consider versioning your config file
5. **Monitor performance** - track how changes affect pattern detection rates

## Integration with Paper Trading

The configuration system is automatically used by the paper trading engine. Changes to pattern detection parameters will immediately affect:

- Signal generation frequency
- Pattern confidence scores
- Trade entry decisions
- Strategy performance metrics

Monitor your paper trading results after configuration changes to ensure the new parameters work well with your strategy. 