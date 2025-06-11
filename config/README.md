# Centralized Configuration System

This directory contains all configurable parameters for the Bistoury trading system in a unified JSON format. All configuration files use the same JSON structure for consistency and ease of management.

## Configuration Files

### 📋 Configuration Structure

```
config/
├── pattern_detection.json  # Pattern detection thresholds and parameters
├── strategy.json          # Strategy-level analysis and trading logic
├── agents.json            # Agent-specific configurations
├── trading.json           # Trading and market data settings
└── README.md              # This documentation
```

## 🔧 Configuration Files Overview

### 1. `pattern_detection.json`
**Pattern Detection Configuration** - Controls how candlestick patterns are detected and scored.

**Key Sections:**
- **Pattern-specific settings**: Thresholds for Doji, Hammer, Engulfing, Harami, etc.
- **Confidence scoring**: How pattern confidence is calculated
- **Reliability parameters**: Pattern reliability calculations
- **Probability adjustments**: Bullish/bearish bias settings

**Example tweaks:**
```json
{
  "engulfing": {
    "min_engulfing_ratio": "1.5",  // How much larger the engulfing body must be
    "base_confidence": "30"        // Starting confidence score
  }
}
```

### 2. `strategy.json`  
**Strategy Configuration** - High-level trading strategy parameters.

**Key Sections:**
- **timeframe_analysis**: Which timeframes to analyze and data quality requirements
- **pattern_detection**: Global pattern detection settings
- **volume_analysis**: Volume confirmation parameters  
- **risk_management**: Stop loss, take profit, risk ratios
- **signal_generation**: Signal creation and expiry settings

**Example tweaks:**
```json
{
  "timeframe_analysis": {
    "data_quality_threshold": 60.0,  // Minimum data quality % required
    "primary_timeframe": "5m"        // Main timeframe for signals
  },
  "pattern_detection": {
    "single_pattern_confidence_threshold": 50.0,  // Single pattern minimum confidence
    "multi_pattern_confidence_threshold": 30.0    // Multi-pattern minimum confidence  
  }
}
```

### 3. `agents.json`
**Agent Configuration** - Settings for individual trading agents.

**Key Sections:**
- **candlestick_strategy**: Candlestick pattern strategy agent settings
- **paper_trading**: Paper trading engine configuration
- **signal_manager**: Signal management and processing

**Example tweaks:**
```json
{
  "candlestick_strategy": {
    "symbols": ["BTC", "ETH", "SOL"],           // Which symbols to trade
    "min_confidence_threshold": 0.60,          // Minimum signal confidence
    "signal_expiry_minutes": 15                // How long signals remain valid
  }
}
```

### 4. `trading.json`
**Trading Configuration** - Market data, execution, and risk management settings.

**Key Sections:**
- **market_data**: Data source settings and connection parameters
- **position_management**: Position sizing and limits
- **risk_management**: Global risk controls
- **execution**: Order execution parameters
- **fees**: Trading fee calculations

**Example tweaks:**
```json
{
  "position_management": {
    "default_position_size_pct": 2.0,    // Default position size as % of capital
    "max_position_size_pct": 10.0        // Maximum position size limit
  },
  "risk_management": {
    "global_stop_loss_pct": 5.0,         // Global stop loss percentage
    "daily_loss_limit_pct": 10.0         // Daily loss limit
  }
}
```

## 🛠️ Using the Configuration System

### Loading Configuration in Code

```python
from bistoury.config_manager import get_config_manager

# Get the global config manager
config = get_config_manager()

# Get specific values
primary_timeframe = config.get('strategy', 'timeframe_analysis', 'primary_timeframe')
min_confidence = config.get_decimal('strategy', 'pattern_detection', 'min_pattern_confidence')
symbols = config.get_list('agents', 'candlestick_strategy', 'symbols')

# Get entire sections
trading_config = config.get_section('trading', 'position_management')
```

### Convenience Functions

```python
from bistoury.config_manager import (
    get_pattern_config,    # Pattern detection values
    get_strategy_config,   # Strategy values  
    get_agent_config,      # Agent values
    get_trading_config     # Trading values
)

# Quick access to specific config types
min_engulfing = get_pattern_config('engulfing', 'min_engulfing_ratio')
data_quality = get_strategy_config('timeframe_analysis', 'data_quality_threshold')
symbols = get_agent_config('candlestick_strategy', 'symbols')
position_size = get_trading_config('position_management', 'default_position_size_pct')
```

### Runtime Configuration Changes

```python
# Change values at runtime (not persisted)
config.set('strategy', ['pattern_detection', 'min_pattern_confidence'], 70)

# Reload from files
config.reload('strategy')  # Reload specific config
config.reload()           # Reload all configs

# Save changes to file
config.save('strategy')
```

## 🎯 Common Configuration Tweaks

### Making Patterns More/Less Sensitive

**More Sensitive (detect more patterns):**
```json
// strategy.json
{
  "pattern_detection": {
    "single_pattern_confidence_threshold": 30.0,  // Lower from 50
    "multi_pattern_confidence_threshold": 20.0     // Lower from 30
  },
  "timeframe_analysis": {
    "data_quality_threshold": 30.0                 // Lower from 60
  }
}
```

**Less Sensitive (detect fewer, higher-quality patterns):**
```json
// strategy.json  
{
  "pattern_detection": {
    "single_pattern_confidence_threshold": 70.0,  // Higher from 50
    "multi_pattern_confidence_threshold": 60.0     // Higher from 30
  },
  "timeframe_analysis": {
    "data_quality_threshold": 80.0                 // Higher from 60
  }
}
```

### Adjusting Risk Management

**More Conservative:**
```json
// trading.json
{
  "risk_management": {
    "global_stop_loss_pct": 2.0,          // Tighter stop loss
    "daily_loss_limit_pct": 5.0,          // Lower daily limit
    "max_drawdown_pct": 10.0              // Stricter drawdown
  },
  "position_management": {
    "default_position_size_pct": 1.0      // Smaller positions
  }
}
```

**More Aggressive:**
```json
// trading.json
{
  "risk_management": {
    "global_stop_loss_pct": 8.0,          // Wider stop loss
    "daily_loss_limit_pct": 20.0,         // Higher daily limit
    "max_drawdown_pct": 30.0              // More drawdown tolerance
  },
  "position_management": {
    "default_position_size_pct": 5.0      // Larger positions
  }
}
```

### Pattern-Specific Adjustments

**Make Engulfing Patterns More Strict:**
```json
// pattern_detection.json
{
  "engulfing": {
    "min_engulfing_ratio": "2.0",         // Require 2x larger body (vs 1.5x)
    "base_confidence": "40",              // Higher base confidence
    "volume_confirmed_score": "30"        // Require volume confirmation
  }
}
```

**Make Doji Patterns More Lenient:**
```json
// pattern_detection.json
{
  "doji": {
    "max_body_ratio": "0.03",             // Allow 3% body (vs 2%)
    "min_range_ratio": "0.003"            // Lower minimum range requirement
  }
}
```

## 🔍 Configuration Validation

The system includes built-in validation:

```python
# Validate configuration
config = get_config_manager()
validation_result = config.validate_config('strategy')

if validation_result['errors']:
    print("❌ Configuration errors:", validation_result['errors'])
if validation_result['warnings']:
    print("⚠️ Configuration warnings:", validation_result['warnings'])
```

## 📁 Migration from Legacy Config

The system automatically falls back to defaults if centralized config is unavailable, ensuring backward compatibility during migration.

**Classes with centralized config support:**
- ✅ `PatternDetectionConfig` - Loads from `pattern_detection.json`
- ✅ `StrategyConfiguration` - Loads from `strategy.json` 
- ✅ `CandlestickStrategyConfig` - Loads from `agents.json`
- ✅ `ConfigManager` - Central configuration management

**Loading centralized config:**
```python
# New centralized approach
strategy_config = StrategyConfiguration.from_config_manager()
agent_config = CandlestickStrategyConfig.from_config_manager()

# Legacy approach (still works as fallback)
strategy_config = StrategyConfiguration()  # Uses defaults
agent_config = CandlestickStrategyConfig()  # Uses defaults
```

## 🎨 Best Practices

1. **Make small incremental changes** - Test one parameter at a time
2. **Document your changes** - Keep notes on what you modified and why
3. **Backup before major changes** - Copy config files before experimentation
4. **Use version control** - Track configuration changes in git
5. **Test in paper trading first** - Validate changes before live trading
6. **Monitor performance impact** - Watch how changes affect signal generation

## 🚨 Important Notes

- **JSON Format Only**: All config files must be valid JSON
- **Decimal Strings**: Numeric values should be strings for precision (e.g., `"1.5"` not `1.5`)
- **Case Sensitive**: All keys are case-sensitive
- **Automatic Fallbacks**: System falls back to defaults if config loading fails
- **Runtime Changes**: Use `config.set()` for temporary changes, `config.save()` to persist

## 📊 Configuration Monitoring

Monitor configuration usage and effectiveness:

```python
# Get current configuration status
config_manager = get_config_manager()
all_configs = config_manager.list_configs()

# Check what configuration is being used
print(f"Strategy data quality threshold: {get_strategy_config('timeframe_analysis', 'data_quality_threshold')}")
print(f"Pattern confidence threshold: {get_strategy_config('pattern_detection', 'min_pattern_confidence')}")
print(f"Agent symbols: {get_agent_config('candlestick_strategy', 'symbols')}")
```

---

**📝 Need Help?** 
- Check the validation system for configuration errors
- Review fallback defaults in the source code
- Test configuration changes in paper trading mode first
- Monitor logs for configuration loading messages 