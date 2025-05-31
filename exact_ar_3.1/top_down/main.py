import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import json
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from config.model_params import ModelParams
from utils.logging_config import setup_logger
from top_down.simulation.driver import run_simulation
from utils.numerical import validate_parameters

# Setup logger
logger = setup_logger(
    name="monte_carlo_sim",
    level="INFO",
    log_file="output/simulation_results.log"
)

def get_user_inputs():
    """Get simulation parameters from user interactively"""
    print("\n=== Monte Carlo Default Simulation ===\n")
    
    # Get number of simulation runs
    while True:
        try:
            runs_input = input("Number of simulation runs [100]: ").strip()
            runs = 100 if runs_input == "" else int(runs_input)
            if runs <= 0:
                print("Number of runs must be positive!")
                continue
            break
        except ValueError:
            print("Please enter a valid integer!")
    
    # Get time horizon
    while True:
        try:
            horizon_input = input("Time horizon in years [5.0]: ").strip()
            horizon = 5.0 if horizon_input == "" else float(horizon_input)
            if horizon <= 0:
                print("Time horizon must be positive!")
                continue
            break
        except ValueError:
            print("Please enter a valid number!")
    
    # Get initial intensity
    while True:
        try:
            lambda0_input = input("Initial intensity [0.7]: ").strip()
            lambda0 = 0.7 if lambda0_input == "" else float(lambda0_input)
            if lambda0 < 0:
                print("Initial intensity cannot be negative!")
                continue
            break
        except ValueError:
            print("Please enter a valid number!")
    
    # Get output directory
    output_input = input("Output directory [output]: ").strip()
    output = "top_down/output" if output_input == "" else output_input
    
    visualize_input = input("Generate visualizations? (y/n) [y]: ").lower().strip()
    visualize = visualize_input == "" or visualize_input.startswith("y")
    
    # Verbose output?
    verbose_input = input("Verbose output? (y/n) [n]: ").lower().strip()
    verbose = verbose_input.startswith("y")
    
    print("\nSimulation Parameters:")
    print(f"- Number of runs: {runs}")
    print(f"- Time horizon: {horizon} years")
    print(f"- Initial intensity: {lambda0}")
    print(f"- Output directory: {output}")
    print(f"- Generate visualizations: {'Yes' if visualize else 'No'}")
    print(f"- Verbose output: {'Yes' if verbose else 'No'}")
    
    confirm = input("\nConfirm these parameters? (y/n) [y]: ").lower().strip()
    if confirm != "" and not confirm.startswith("y"):
        print("Simulation cancelled by user.")
        sys.exit(0)
    
    return {
        "runs": runs,
        "horizon": horizon,
        "lambda0": lambda0,
        "output": output,
        "visualize": visualize,
        "verbose": verbose
    }

def setup_output_directory(output_dir: str):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path

def create_model_params() -> ModelParams:
    """Define model parameters for the simulation"""
    return ModelParams(
        kappa=2.62,    # Mean reversion speed
        theta=1.61,    # Long-term mean
        sigma=0.62,    # Volatility
        gamma=0.99     # Jump size multiplier (paper uses 2.99 but we use 0.99 for stability)
    )

def run_multiple_simulations(
    n_runs: int, 
    lambda_0: float, 
    T_max: float, 
    model_params: ModelParams
) -> List[List[Tuple[float, float, float]]]:
    """Run multiple simulation iterations"""
    all_results = []
    start_time = time.time()
    
    for i in range(n_runs):
        if i % 10 == 0:
            logger.info(f"Starting simulation run {i+1}/{n_runs}")
            print(f"Progress: {i+1}/{n_runs} simulations completed", end="\r")
        
        result = run_simulation(
            lambda_0=lambda_0,
            T_max=T_max,
            model_params=model_params
        )
        all_results.append(result)
        
    elapsed = time.time() - start_time
    logger.info(f"Completed {n_runs} simulations in {elapsed:.2f} seconds")
    print(f"\nCompleted {n_runs} simulations in {elapsed:.2f} seconds")
    
    return all_results

def compute_statistics(results: List[List[Tuple[float, float, float]]], T_max: float):
    n_defaults = [len(sim) for sim in results]
    
    total_losses = [sum(loss for _, _, loss in sim) for sim in results]
    
    max_intensities = [max([intensity for _, intensity, _ in sim] if sim else [0]) for sim in results]
    
    # Default times (flatten and sort)
    all_default_times = [t for sim in results for t, _, _ in sim if t <= T_max]
    all_default_times.sort()
    
    # Create statistics dictionary
    stats = {
        "n_simulations": len(results),
        "time_horizon": T_max,
        "mean_defaults": np.mean(n_defaults),
        "std_defaults": np.std(n_defaults),
        "mean_loss": np.mean(total_losses),
        "std_loss": np.std(total_losses),
        "mean_max_intensity": np.mean(max_intensities),
        "std_max_intensity": np.std(max_intensities),
        "min_defaults": min(n_defaults),
        "max_defaults": max(n_defaults),
        "min_loss": min(total_losses),
        "max_loss": max(total_losses),
    }
    
    return stats, all_default_times, total_losses

def create_visualizations(
    results: List[List[Tuple[float, float, float]]], 
    stats: Dict, 
    default_times: List[float], 
    losses: List[float],
    output_dir: Path
):
    """Create various visualizations of the simulation results"""
    print("Creating visualizations...")
    
    # Set style for plots
    plt.style.use('seaborn-v0_8-darkgrid')
    
    # Figure 1: Histogram of number of defaults
    n_defaults = [len(sim) for sim in results]
    plt.figure(figsize=(10, 6))
    plt.hist(n_defaults, bins=max(10, max(n_defaults) // 2), alpha=0.7, color='blue')
    plt.axvline(stats["mean_defaults"], color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {stats["mean_defaults"]:.2f}')
    plt.xlabel('Number of Defaults')
    plt.ylabel('Frequency')
    plt.title('Distribution of Number of Defaults')
    plt.legend()
    plt.savefig(output_dir / 'defaults_histogram.png', dpi=300, bbox_inches='tight')
    
    # Figure 2: Histogram of total losses
    plt.figure(figsize=(10, 6))
    plt.hist(losses, bins=20, alpha=0.7, color='blue')
    plt.axvline(stats["mean_loss"], color='red', linestyle='dashed', linewidth=2, 
                label=f'Mean: {stats["mean_loss"]:.2f}')
    plt.xlabel('Total Loss')
    plt.ylabel('Frequency')
    plt.title('Distribution of Total Losses')
    plt.legend()
    plt.savefig(output_dir / 'losses_histogram.png', dpi=300, bbox_inches='tight')
    
    # Figure 3: Default time density
    if default_times:
        plt.figure(figsize=(10, 6))
        plt.hist(default_times, bins=30, alpha=0.7, density=True, color='blue')
        plt.xlabel('Default Time')
        plt.ylabel('Density')
        plt.title('Distribution of Default Times')
        plt.savefig(output_dir / 'default_times_density.png', dpi=300, bbox_inches='tight')
    
    # Figure 4: Sample paths of intensity
    plt.figure(figsize=(12, 7))
    for i in range(max(2, len(results))):
        if results[i]:
            times = [0] + [t for t, _, _ in results[i]]
            intensities = [results[i][0][1] - results[i][0][2] * 0.99] + [intensity for _, intensity, _ in results[i]]
            plt.step(times, intensities, where='post', label=f'Simulation {i+1}')
    
    plt.xlabel('Time')
    plt.ylabel('Intensity')
    plt.title('Sample Paths of Default Intensity')
    plt.legend()
    plt.grid(True)
    plt.savefig(output_dir / 'intensity_sample_paths.png', dpi=300, bbox_inches='tight')
    
    plt.close('all')
    print(f"Visualizations saved to {output_dir}")

def save_results(stats: Dict, output_dir: Path):
    """Save simulation statistics to files"""
    print("Saving results...")
    # Save statistics as JSON
    with open(output_dir / 'statistics.json', 'w') as f:
        json.dump(stats, f, indent=4)
    
    with open(output_dir / 'summary.txt', 'w') as f:
        f.write("Monte Carlo Default Simulation Summary\n")
        f.write("=====================================\n\n")
        f.write(f"Number of simulations: {stats['n_simulations']}\n")
        f.write(f"Time horizon: {stats['time_horizon']} years\n\n")
        
        f.write("Default Statistics:\n")
        f.write(f"  Mean number of defaults: {stats['mean_defaults']:.4f} ± {stats['std_defaults']:.4f}\n")
        f.write(f"  Range: [{stats['min_defaults']}, {stats['max_defaults']}]\n")
        
        f.write("\nLoss Statistics:\n")
        f.write(f"  Mean total loss: {stats['mean_loss']:.4f} ± {stats['std_loss']:.4f}\n")
        f.write(f"  Range: [{stats['min_loss']:.4f}, {stats['max_loss']:.4f}]\n")
        
        f.write("\nIntensity Statistics:\n")
        f.write(f"  Mean max intensity: {stats['mean_max_intensity']:.4f} ± {stats['std_max_intensity']:.4f}\n")
    
    print(f"Results saved to {output_dir}/summary.txt")

def main():
    """Main function"""
    params = get_user_inputs()
    
    output_dir = setup_output_directory(params["output"])
    
    # Set log level
    if params["verbose"]:
        logger.setLevel("DEBUG")

    model_params = create_model_params()
    validate_parameters(model_params, logger)
    
    print(f"\nStarting Monte Carlo simulation with {params['runs']} runs")
    print(f"Time horizon: {params['horizon']} years, Initial intensity: {params['lambda0']}")
    
    results = run_multiple_simulations(
        n_runs=params["runs"],
        lambda_0=params["lambda0"],
        T_max=params["horizon"],
        model_params=model_params
    )
    
    stats, default_times, total_losses = compute_statistics(results, params["horizon"])
    
    print("\nSimulation Results Summary:")
    print(f"Mean defaults: {stats['mean_defaults']:.4f} ± {stats['std_defaults']:.4f}")
    print(f"Mean total loss: {stats['mean_loss']:.4f} ± {stats['std_loss']:.4f}")
    
    if params["visualize"]:
        create_visualizations(results, stats, default_times, total_losses, output_dir)
    
    save_results(stats, output_dir)
    print("\nSimulation complete!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user.")
    except Exception as e:
        print(f"\nError: {e}")
        logger.exception("Unhandled exception in main")