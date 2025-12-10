# RJMCMC Nonlinear Survival Models

Julia implementation of reversible jump MCMC survival models for simulations and real-world analyses (German Breast Cancer and Primary Biliary Cirrhosis).

## Features

- **Three RJMCMC Methods**: 
  - NonLinear1: Default uniform order statistics prior
  - NonLinear2: Dirichlet-Gamma prior for knot locations
  - CoxPH: Baseline Cox proportional hazards model
- **Parallel Execution**: Thread-based parallelism with ordered execution (by g_type, then n_values)
- **Resume Support**: Automatic checkpointing and resume from existing results
- **Progress Tracking**: Real-time progress bar for long-running simulations
- **Comprehensive Outputs**: Posterior summaries, IBS metrics, and publication-ready plots

## Project Structure

- `config.jl`: Central defaults for simulations (`demo`/`full`) and real-data runs
- `model.jl`: RJMCMC algorithms and utilities shared by all scripts
- `simulation.jl`: End-to-end simulation pipeline with caching, resume support, progress bar, and thread-based parallelism
- `real_data_GBC.jl` / `real_data_PBC.jl`: Analyses for the GBC and PBC datasets in `GermanBC/` and `PBC/`
- `install_packages.jl`: Script to install all required Julia packages
- `boxplot.R`: R script for generating IBS boxplots (requires ggplot2)
- `results/`: Generated outputs (gitignored)

## Installation

### Prerequisites

- **Julia ≥1.9**: Download from [julialang.org](https://julialang.org/downloads/)
- **R** (optional, for boxplot generation): Download from [r-project.org](https://www.r-project.org/)

### Install Julia Packages

**Recommended method** (easiest):
```bash
julia install_packages.jl
```

**Alternative methods**:

1. **Windows PowerShell**:
   ```powershell
   julia -e "using Pkg; Pkg.add([\"DataFrames\", \"CSV\", \"Distributions\", \"ProgressMeter\", \"StatsBase\", \"VectorizedStatistics\", \"MLDataUtils\", \"Plots\", \"StatsPlots\", \"CategoricalArrays\", \"SpecialFunctions\", \"LaTeXStrings\"])"
   ```

2. **Interactive Julia REPL**:
```julia
   julia
   ] add DataFrames CSV Distributions ProgressMeter StatsBase VectorizedStatistics MLDataUtils Plots StatsPlots CategoricalArrays SpecialFunctions LaTeXStrings
   ```
   (Press `Ctrl+C` to exit package mode, then type `exit()` to quit)

### Install R Package (for boxplots)

```r
Rscript -e "install.packages('ggplot2', repos='https://cran.rstudio.com/')"
```

Or in R console:
```r
install.packages("ggplot2")
```

## Running Simulations

### Basic Usage

```bash
# Quick demo run (all scenarios: n=200,400,800, 5 replications each)
julia simulation.jl --demo

# Full run (all scenarios: n=200,400,800, 1000 replications each)
julia simulation.jl --full

# Use specific number of threads
JULIA_NUM_THREADS=32 julia simulation.jl --full

# Windows PowerShell
$env:JULIA_NUM_THREADS=32; julia simulation.jl --full
```

### Command-Line Options

- `--demo`: Run demo mode (all scenarios with 5 replications each)
- `--full`: Run full mode (all scenarios with 1000 replications each)
- `--reset`: Clear existing checkpoints and rerun from scratch
- `--replot`: Regenerate plots from existing results
- `--plot-only`: Only generate plots without running simulations
- `--workers=N`: Specify number of worker threads (default: all available threads)

### Execution Order

The simulation executes tasks in a specific order:
1. **By g_type**: `linear` → `quad` → `sin`
2. **By n_values**: Within each g_type, tasks are processed by sample size (200 → 400 → 800)
3. **By replication**: Within each (g_type, n) combination, replications are executed in parallel but tasks are sorted by replication index (1, 2, ..., N)

This ensures that all `linear` scenarios complete before starting `quad`, and all `quad` complete before starting `sin`.

### Simulation Configuration

**Demo Mode** (`--demo`):
- Sample sizes: `[200, 400, 800]`
- Replications: `5` per scenario
- Total tasks: 3 g_types × 3 n_values × 5 replications = **45 tasks**

**Full Mode** (`--full`):
- Sample sizes: `[200, 400, 800]`
- Replications: `1000` per scenario
- Total tasks: 3 g_types × 3 n_values × 1000 replications = **9000 tasks**

### Checkpoints and Resume

- Checkpoints are saved under `results/simulation/<mode>/g=<type>/n=<size>/rep=<id>/results_dict.jls`
- Rerunning will automatically skip finished tasks and continue from existing checkpoints
- Use `--reset` to clear all checkpoints and start fresh

### Output Files

Results are saved in `results/simulation/<mode>/`:
- `simu_summary.csv`: Summary statistics for all methods
- `df_IBS.csv`: Integrated Brier Score (IBS) metrics
- `plots/`: Individual plots for each (g_type, n) combination
- `plots_manuscript/`: Publication-ready figures (Figures 1-4 style)

## Real Data Analysis

```bash
# German Breast Cancer dataset
julia real_data_GBC.jl

# Primary Biliary Cirrhosis dataset
julia real_data_PBC.jl
```

Outputs (posterior summaries, estimated g(z), IBS averages) are written to `results/real_data/gbc/` and `results/real_data/pbc/`.

## Dependencies

See `requirements.txt` for a complete list of required packages:
- **Julia packages**: DataFrames, CSV, Distributions, ProgressMeter, StatsBase, VectorizedStatistics, MLDataUtils, Plots, StatsPlots, CategoricalArrays, SpecialFunctions, LaTeXStrings
- **R packages**: ggplot2 (for boxplot generation)

## Notes

- The simulation uses thread-based parallelism. Set `JULIA_NUM_THREADS` environment variable to control the number of threads.
- On Windows PowerShell, use `$env:JULIA_NUM_THREADS=N` to set thread count.
- The code automatically detects available threads and warns if requested threads exceed available threads.
- For best performance, set `JULIA_NUM_THREADS` to match your CPU core count.
