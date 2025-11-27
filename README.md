# RJMCMC Nonlinear Survival Models

Julia implementation of reversible jump MCMC survival models for simulations and real-world analyses (German Breast Cancer and Primary Biliary Cirrhosis).

## Layout
- `config.jl`: central defaults for simulations (`demo`/`full`) and real-data runs.
- `model.jl`: RJMCMC algorithms and utilities shared by all scripts.
- `simulation.jl`: end-to-end simulation pipeline with caching, resume support, progress bar, and thread-based parallelism.
- `real_data_GBC.jl` / `real_data_PBC.jl`: analyses for the GBC and PBC datasets in `GermanBC/` and `PBC/`.
- `results/`: generated outputs (gitignored).

## Dependencies
Install Julia ≥1.9 and add the packages listed in `requirements.txt` using Julia's package manager:

```julia
import Pkg; Pkg.add(readlines("requirements.txt"))
```

## Running simulations
```bash
julia --threads=4 simulation.jl --demo      # quick smoke test
julia --threads=8 simulation.jl --full      # full run matching the notebook defaults
julia simulation.jl --reset --full          # clear checkpoints in results/simulation/ and rerun
```

- Checkpoints live under `results/simulation/<mode>/g=<type>/n=<size>/rep=<id>/results_dict.jls`.
- Rerunning will skip finished tasks and continue from existing checkpoints automatically.
- Summary files: `simu_summary.csv` and `df_IBS.csv` in the same `results/simulation/<mode>/` folder.

## Real data
```bash
julia real_data_GBC.jl
julia real_data_PBC.jl
```

Outputs (posterior summaries, estimated \(g(z)\), IBS averages) are written to `results/real_data/gbc/` and `results/real_data/pbc/`.
