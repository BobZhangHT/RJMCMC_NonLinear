module Config

# Centralized configuration structures and defaults shared by simulations and real-data scripts.

export SimulationMode, SimulationConfig, default_simulation_config, RealDataConfig, real_data_config, RESULTS_DIR

using Base.Threads

# Location to store all generated artifacts (results/ is gitignored).
# Anchor to repository directory so running from other working dirs still works.
const RESULTS_DIR = joinpath(@__DIR__, "results")

@enum SimulationMode begin
    demo
    full
end

struct SimulationConfig
    mode::SimulationMode
    ns::Int
    burn_in::Int
    n_workers::Int
    n_test::Int
    replications::Int
    g_types::Vector{String}
    n_values::Vector{Int}
    data_seed::Int
    mcmc_seed::Int
    hazard_a::Float64
    hazard_b::Float64
    z_min::Float64
    z_max::Float64
    t_grid::Vector{Float64}
    z_grid::Vector{Float64}
end

struct RealDataConfig
    ns::Int
    burn_in::Int
    mcmc_seed::Int
    folds::Int
end

"""
    default_simulation_config(mode=full; n_workers=max(1, Threads.nthreads()))

Return a ready-to-use simulation configuration. `demo` keeps the same sampler settings
as the notebooks but uses fewer replications; `full` mirrors the original defaults.
"""
function default_simulation_config(mode::SimulationMode=full; n_workers::Int=max(1, Threads.nthreads()))
    ns = 10000
    n_test = 500
    g_types = ["linear", "quad", "sin"]
    # Demo: all manuscript sizes but with fewer replications; Full: all manuscript sizes.
    n_values = [200, 400, 800]
    # Use a light footprint for demo, full manuscript settings use 1000 reps.
    replications = mode == demo ? 5 : 1000

    burn_in = ns ÷ 2
    SimulationConfig(
        mode,
        ns,
        burn_in,
        n_workers,
        n_test,
        replications,
        g_types,
        n_values,
        2024,
        2024,
        0.5,
        2.0,
        0.0,
        2 * pi,
        collect(range(0; stop=3, length=100)),
        collect(range(0; stop=2 * pi, length=100)),
    )
end

"""
    real_data_config(; ns=20_000, burn_in=10_000, mcmc_seed=2024, folds=10)

Standard settings for the real-data analyses.
"""
function real_data_config(; ns::Int=20_000, burn_in::Int=10_000, mcmc_seed::Int=2024, folds::Int=10)
    RealDataConfig(ns, burn_in, mcmc_seed, folds)
end

end # module
