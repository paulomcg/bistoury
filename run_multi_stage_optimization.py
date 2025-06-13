#!/usr/bin/env python3
"""
Automated multi-stage optimization script.
Each stage automatically feeds into the next with progressively narrowed search spaces.
"""

import subprocess
import time
import json
from pathlib import Path
from analyze_optimization_stages import analyze_stage_results, generate_next_stage_config, save_stage_config

def run_optimization_stage(config_file: str, study_name: str, trials: int, jobs: int = 8) -> bool:
    """Run a single optimization stage and return success status."""
    
    print(f"\n🚀 Starting {study_name}")
    print(f"Config: {config_file}")
    print(f"Trials: {trials}, Jobs: {jobs}")
    print("-" * 50)
    
    cmd = [
        "python", "-m", "bistoury", "optimize",
        f"--trials={trials}",
        f"--jobs={jobs}",
        f"--study-name={study_name}",
        f"--config={config_file}"
    ]
    
    start_time = time.time()
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        duration = time.time() - start_time
        
        print(f"✅ {study_name} completed successfully in {duration/60:.1f} minutes")
        print(f"Output: {result.stdout[-500:]}")  # Last 500 chars
        return True
        
    except subprocess.CalledProcessError as e:
        duration = time.time() - start_time
        print(f"❌ {study_name} failed after {duration/60:.1f} minutes")
        print(f"Error: {e.stderr[-500:]}")  # Last 500 chars of error
        return False

def run_complete_multi_stage_optimization():
    """Run the complete multi-stage optimization pipeline."""
    
    print("🎯 Multi-Stage Optimization Pipeline")
    print("=" * 60)
    print("Stage 1: Broad Exploration (500 trials)")
    print("Stage 2: Focused Refinement (300 trials)")  
    print("Stage 3: Final Optimization (200 trials)")
    print("=" * 60)
    
    # Stage 1: Broad Exploration
    stage1_success = run_optimization_stage(
        config_file="config/strategy.json",
        study_name="stage1_explore", 
        trials=500,
        jobs=8
    )
    
    if not stage1_success:
        print("❌ Stage 1 failed, aborting multi-stage optimization")
        return False
    
    # Analyze Stage 1 and generate Stage 2 config
    print("\n📊 Analyzing Stage 1 results...")
    try:
        stage1_analysis = analyze_stage_results("stage1_explore", "./backtest_results")
        stage2_config = generate_next_stage_config(stage1_analysis, "stage2_refine", narrowing_factor=0.4)
        save_stage_config(stage2_config, "config/strategy_stage2.json")
    except Exception as e:
        print(f"❌ Failed to analyze Stage 1: {e}")
        return False
    
    # Stage 2: Focused Refinement
    stage2_success = run_optimization_stage(
        config_file="config/strategy_stage2.json",
        study_name="stage2_refine",
        trials=300,
        jobs=8
    )
    
    if not stage2_success:
        print("❌ Stage 2 failed, skipping Stage 3")
        return False
    
    # Analyze Stage 2 and generate Stage 3 config
    print("\n📊 Analyzing Stage 2 results...")
    try:
        stage2_analysis = analyze_stage_results("stage2_refine", "./backtest_results")
        stage3_config = generate_next_stage_config(stage2_analysis, "stage3_final", narrowing_factor=0.6)
        save_stage_config(stage3_config, "config/strategy_stage3.json")
    except Exception as e:
        print(f"❌ Failed to analyze Stage 2: {e}")
        return False
    
    # Stage 3: Final Optimization
    stage3_success = run_optimization_stage(
        config_file="config/strategy_stage3.json",
        study_name="stage3_final",
        trials=200,
        jobs=8
    )
    
    if stage3_success:
        print("\n🎉 Multi-stage optimization completed successfully!")
        print("\n📈 Final Results Summary:")
        
        # Show progression across stages
        for stage_name in ["stage1_explore", "stage2_refine", "stage3_final"]:
            try:
                analysis = analyze_stage_results(stage_name, "./backtest_results")
                print(f"  {stage_name}: {analysis['best_value']:.6f} (best)")
            except:
                print(f"  {stage_name}: Failed to load")
        
        return True
    else:
        print("❌ Stage 3 failed")
        return False

def run_parallel_multi_stage():
    """Run multiple independent multi-stage optimizations in parallel."""
    
    print("🚀 Parallel Multi-Stage Optimization")
    print("Running 3 independent multi-stage pipelines simultaneously")
    
    import concurrent.futures
    import threading
    
    def run_pipeline(pipeline_id: int):
        """Run a single pipeline with unique study names."""
        
        # Modify study names to be unique per pipeline
        stages = [
            ("config/strategy.json", f"pipeline{pipeline_id}_stage1", 200),
            (f"config/strategy_p{pipeline_id}_stage2.json", f"pipeline{pipeline_id}_stage2", 150),
            (f"config/strategy_p{pipeline_id}_stage3.json", f"pipeline{pipeline_id}_stage3", 100)
        ]
        
        print(f"\n🔄 Starting Pipeline {pipeline_id}")
        
        for i, (config_file, study_name, trials) in enumerate(stages):
            if i > 0:  # Generate config for stages 2 and 3
                prev_study = stages[i-1][1]
                try:
                    analysis = analyze_stage_results(prev_study, "./backtest_results")
                    narrowing = 0.4 if i == 1 else 0.6
                    next_config = generate_next_stage_config(analysis, study_name, narrowing)
                    save_stage_config(next_config, config_file)
                except Exception as e:
                    print(f"❌ Pipeline {pipeline_id} failed at stage {i+1}: {e}")
                    return False
            
            success = run_optimization_stage(config_file, study_name, trials, jobs=4)
            if not success:
                print(f"❌ Pipeline {pipeline_id} failed at stage {i+1}")
                return False
        
        print(f"✅ Pipeline {pipeline_id} completed successfully!")
        return True
    
    # Run 3 pipelines in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(run_pipeline, i) for i in range(1, 4)]
        results = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    successful_pipelines = sum(results)
    print(f"\n🎯 Parallel optimization complete: {successful_pipelines}/3 pipelines succeeded")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--parallel":
        run_parallel_multi_stage()
    else:
        run_complete_multi_stage_optimization() 