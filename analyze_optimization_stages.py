#!/usr/bin/env python3
"""
Multi-stage optimization analysis and configuration script.
Analyzes results from previous stages to configure the next stage.
"""

import optuna
import json
import numpy as np
from pathlib import Path

def analyze_stage_results(study_name: str, storage_dir: str = "./backtest_results") -> dict:
    """Analyze results from a completed optimization stage."""
    
    storage_path = Path(storage_dir) / f"{study_name}.db"
    if not storage_path.exists():
        raise FileNotFoundError(f"Study database not found: {storage_path}")
    
    storage_url = f"sqlite:///{storage_path}"
    study = optuna.load_study(study_name=study_name, storage=storage_url)
    
    print(f"\n📊 Analyzing {study_name}")
    print(f"Total trials: {len(study.trials)}")
    print(f"Best value: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    
    # Get top 10% of trials
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value or -float('inf'), reverse=True)
    top_10_percent = completed_trials[:max(1, len(completed_trials) // 10)]
    
    print(f"\nTop 10% trials: {len(top_10_percent)}")
    
    # Analyze parameter ranges in top trials
    param_analysis = {}
    for param_name in study.best_params.keys():
        values = [t.params[param_name] for t in top_10_percent if param_name in t.params]
        if values:
            param_analysis[param_name] = {
                'min': min(values),
                'max': max(values),
                'mean': np.mean(values),
                'std': np.std(values),
                'q25': np.percentile(values, 25),
                'q75': np.percentile(values, 75)
            }
    
    return {
        'study_name': study_name,
        'total_trials': len(study.trials),
        'best_value': study.best_value,
        'best_params': study.best_params,
        'top_trials_count': len(top_10_percent),
        'param_analysis': param_analysis
    }

def generate_next_stage_config(analysis: dict, stage_name: str, narrowing_factor: float = 0.3) -> dict:
    """Generate configuration for the next optimization stage based on analysis."""
    
    print(f"\n🎯 Generating config for {stage_name}")
    print(f"Narrowing factor: {narrowing_factor} (0.0 = no change, 1.0 = point estimate)")
    
    next_config = {
        "tunable_parameters": {},
        "duration": 300,
        "replay_speed": 50.0,
        "timeframe": "1m",
        "symbol": "BTC",
        "initial_balance": 10000
    }
    
    for param_name, stats in analysis['param_analysis'].items():
        # Calculate narrowed range based on top performers
        current_range = stats['max'] - stats['min']
        narrowed_range = current_range * (1 - narrowing_factor)
        
        # Center around the mean of top performers
        center = stats['mean']
        new_min = max(stats['min'], center - narrowed_range / 2)
        new_max = min(stats['max'], center + narrowed_range / 2)
        
        # Ensure minimum range for continued exploration
        min_range = current_range * 0.1  # At least 10% of original range
        if (new_max - new_min) < min_range:
            new_min = center - min_range / 2
            new_max = center + min_range / 2
        
        # Determine parameter type
        param_type = "float"
        if param_name in ["min_candles_required"]:
            param_type = "int"
            new_min = int(new_min)
            new_max = int(new_max)
        
        next_config["tunable_parameters"][param_name] = {
            "min": new_min,
            "max": new_max,
            "type": param_type,
            "description": f"Narrowed from stage analysis (was {stats['min']:.2f}-{stats['max']:.2f})"
        }
        
        print(f"  {param_name}: {stats['min']:.2f}-{stats['max']:.2f} → {new_min:.2f}-{new_max:.2f}")
    
    return next_config

def save_stage_config(config: dict, filename: str):
    """Save the generated configuration for the next stage."""
    with open(filename, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"\n💾 Saved configuration to {filename}")

def run_multi_stage_analysis():
    """Run complete multi-stage analysis and configuration generation."""
    
    print("🚀 Multi-Stage Optimization Analysis")
    print("=" * 50)
    
    # Stage 1 → Stage 2
    try:
        stage1_analysis = analyze_stage_results("stage1_explore")
        stage2_config = generate_next_stage_config(stage1_analysis, "stage2_refine", narrowing_factor=0.4)
        save_stage_config(stage2_config, "config/strategy_stage2.json")
    except FileNotFoundError:
        print("⚠️  Stage 1 results not found, skipping Stage 2 config generation")
    
    # Stage 2 → Stage 3
    try:
        stage2_analysis = analyze_stage_results("stage2_refine")
        stage3_config = generate_next_stage_config(stage2_analysis, "stage3_final", narrowing_factor=0.6)
        save_stage_config(stage3_config, "config/strategy_stage3.json")
    except FileNotFoundError:
        print("⚠️  Stage 2 results not found, skipping Stage 3 config generation")
    
    print("\n✅ Multi-stage analysis complete!")
    print("\nNext steps:")
    print("1. Review generated configs in config/strategy_stage*.json")
    print("2. Run next stage: python -m bistoury optimize --config=config/strategy_stage2.json")

if __name__ == "__main__":
    run_multi_stage_analysis() 