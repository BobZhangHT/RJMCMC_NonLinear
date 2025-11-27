#!/usr/bin/env julia

# End-to-end simulation driver: data generation, RJMCMC fitting, caching,
# parallel execution with progress bar, and summary extraction.

using Random
using Distributions
using DataFrames
using CSV
using Serialization
using ProgressMeter
using StatsBase
using Statistics
using Base.Threads

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "model.jl"))

using .Config
using .RJMCMCModel

struct SimulationTask
    g_type::String  # true g(z) shape
    n::Int          # sample size
    idx::Int        # replication id
end

# Parse command-line flags (--demo/--full/--reset/--workers=N).
function parse_args()
    mode = SimulationMode.full
    reset = false
    workers = nthreads()

    for arg in ARGS
        if arg == "--demo"
            mode = SimulationMode.demo
        elseif arg == "--full"
            mode = SimulationMode.full
        elseif arg == "--reset"
            reset = true
        elseif startswith(arg, "--workers=")
            workers = parse(Int, split(arg, "=")[2])
        end
    end

    return mode, reset, workers
end

# Synthetic survival data generator mirroring the notebook setup.
function data_gen(random_seed; g_type::String="sin", n::Int=200, hazard_a::Float64=0.5, hazard_b::Float64=2.0, z_min::Float64=0.0, z_max::Float64=2pi)
    betas0 = [0.5 0.5]
    lambda_inv_fun = u -> (u / hazard_a)^(1 / hazard_b)

    g_fun = begin
        if g_type == "sin"
            z -> sin.(z)
        elseif g_type == "quad"
            z -> -z .* (z .- 2pi) / 5
        elseif g_type == "U-quad"
            z -> z .* (z .- 2pi) / 5
        else
            z -> 1 ./ (1 .+ exp.(2 .* (pi .- z)))
        end
    end

    X1_dist = Bernoulli(0.5)
    X2_dist = Normal(0, 1)
    Z_dist = Uniform(z_min, z_max)
    C_dist = Exponential(4)

    Random.seed!(random_seed)
    X1 = rand(X1_dist, n)
    X2 = rand(X2_dist, n)
    X_mat = [X1 X2]
    Z = rand(Z_dist, n)

    U = rand(n)
    T = lambda_inv_fun.(-log.(1 .- U) ./ exp.((X_mat * betas0')[:, 1] .+ g_fun(Z)))
    C = min.(rand(C_dist, n), 4)
    Y = min.(T, C)
    Delta = T .< C
    cen_rate = 1 - mean(Delta)

    DataFrame(
        Y=Y,
        Delta=Delta,
        T=T,
        C=C,
        X1=X1,
        X2=X2,
        Z=Z,
        cen_rate=cen_rate,
    )
end

function ensure_results_dir(base_dir::String)
    if !isdir(base_dir)
        mkpath(base_dir)
    end
end

function task_dir(base_dir::String, task::SimulationTask)
    joinpath(base_dir, "g=$(task.g_type)", "n=$(task.n)", "rep=$(task.idx)")
end

function run_single_task(task::SimulationTask, cfg::SimulationConfig, base_dir::String)
    dir = task_dir(base_dir, task)
    results_path = joinpath(dir, "results_dict.jls")
    if isfile(results_path)
        return :skipped
    end

    mkpath(dir)
    train_path = joinpath(dir, "dat_train.csv")
    test_path = joinpath(dir, "dat_test.csv")

    if !isfile(train_path) || !isfile(test_path)
        df_train = data_gen(task.idx; g_type=task.g_type, n=task.n, hazard_a=cfg.hazard_a, hazard_b=cfg.hazard_b, z_min=cfg.z_min, z_max=cfg.z_max)
        df_test = data_gen(task.idx + cfg.data_seed; g_type=task.g_type, n=cfg.n_test, hazard_a=cfg.hazard_a, hazard_b=cfg.hazard_b, z_min=cfg.z_min, z_max=cfg.z_max)
        CSV.write(train_path, df_train)
        CSV.write(test_path, df_test)
    end

    df_train = CSV.read(train_path, DataFrame)
    df_test = CSV.read(test_path, DataFrame)

    Y_train, Delta_train = df_train.Y, df_train.Delta
    X_train = Matrix(df_train[:, [:X1, :X2]])
    Z_train = df_train.Z

    Y_test, Delta_test = df_test.Y, df_test.Delta
    X_test = Matrix(df_test[:, [:X1, :X2]])
    Z_test = df_test.Z

    a, b = minimum(Z_train), maximum(Z_train)

    results_nonlinear = RJMCMC_Nonlinear(X_train, Z_train, Delta_train, Y_train, a, b, cfg.mcmc_seed; ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX, Kmax=DEFAULT_KMAX)
    results_coxph = RJMCMC_CoxPH(X_train, Z_train, Delta_train, Y_train, cfg.mcmc_seed; ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX)

    ns = results_nonlinear["ns"]
    bi = results_nonlinear["burn_in"]
    t_grid = cfg.t_grid
    z_grid = cfg.z_grid

    K_all, zetas_all, xis_all = results_nonlinear["K"], results_nonlinear["zetas"], results_nonlinear["xis"]
    H_all, taus_all, gammas_all = results_nonlinear["H"], results_nonlinear["taus"], results_nonlinear["gammas"]
    Tmax = results_nonlinear["Tmax"]

    g_pos = zeros(length(z_grid))
    for i in (bi + 1):ns
        K = Int(K_all[i])
        zetas = zetas_all[1:K, i]
        xis = xis_all[1:(K + 2), i]
        g_pos .+= g_fun_est(z_grid, a, b, zetas, xis)
    end
    g_pos ./= (ns - bi)

    Lambda_nonlinear = zeros(length(t_grid))
    for i in (bi + 1):ns
        H = Int(H_all[i])
        taus = taus_all[1:H, i]
        gammas = gammas_all[1:(H + 2), i]
        Lambda_nonlinear .+= Lambda_fun_est(t_grid, Tmax, taus, gammas)
    end
    Lambda_nonlinear ./= (ns - bi)

    H_cox_all, taus_cox_all, gammas_cox_all = results_coxph["H"], results_coxph["taus"], results_coxph["gammas"]
    Lambda_coxph = zeros(length(t_grid))
    for i in (bi + 1):ns
        H = Int(H_cox_all[i])
        taus = taus_cox_all[1:H, i]
        gammas = gammas_cox_all[1:(H + 2), i]
        Lambda_coxph .+= Lambda_fun_est(t_grid, Tmax, taus, gammas)
    end
    Lambda_coxph ./= (ns - bi)

    IBS_coxph = IBS(Y_train, Delta_train, X_train, Z_train, Y_test, Delta_test, X_test, Z_test, results_coxph, "coxph")
    IBS_nonlinear = IBS(Y_train, Delta_train, X_train, Z_train, Y_test, Delta_test, X_test, Z_test, results_nonlinear, "nonlinear")

    results_dict = Dict(
        "beta_coxph" => results_coxph["betas"],
        "beta_nonlinear" => results_nonlinear["betas"],
        "IBS_coxph" => IBS_coxph,
        "IBS_nonlinear" => IBS_nonlinear,
        "Lambda_coxph" => Lambda_coxph,
        "Lambda_nonlinear" => Lambda_nonlinear,
        "g_nonlinear" => g_pos,
        "Tmax" => Tmax,
        "Hseq" => results_nonlinear["H"][bi+1:ns],
        "Kseq" => results_nonlinear["K"][bi+1:ns],
        "Hseq_coxph" => results_coxph["H"][bi+1:ns],
        "ns" => ns,
        "burn_in" => bi,
        "mode" => String(cfg.mode),
    )

    open(results_path, "w") do file
        serialize(file, results_dict)
    end

    return :done
end

# Aggregate posterior summaries and IBS metrics across completed replications.
function summarize_tasks(cfg::SimulationConfig, base_dir::String)
    df_summary = DataFrame(
        G_type=String[],
        N=Int[],
        Method=String[],
        Bias1=Float64[],
        M_SE1=Float64[],
        ESD1=Float64[],
        CR1=Float64[],
        MSE1=Float64[],
        Bias2=Float64[],
        M_SE2=Float64[],
        ESD2=Float64[],
        CR2=Float64[],
        MSE2=Float64[],
        H=Float64[],
        K=Float64[],
    )

    df_IBS_summary = DataFrame(N=Int[], G_type=String[], Method=String[], IBS=Float64[])

    for g_type in cfg.g_types
        for n in cfg.n_values
            betas_nonlinear_mat = Float64[]
            betas_se_nonlinear_mat = Float64[]
            betas_nonlinear_mat2 = Float64[]
            betas_se_nonlinear_mat2 = Float64[]
            H_nonlinear = Float64[]
            K_nonlinear = Float64[]
            cr1_nonlinear = Float64[]
            cr2_nonlinear = Float64[]

            betas_coxph_mat = Float64[]
            betas_se_coxph_mat = Float64[]
            betas_coxph_mat2 = Float64[]
            betas_se_coxph_mat2 = Float64[]
            H_coxph = Float64[]
            cr1_coxph = Float64[]
            cr2_coxph = Float64[]

            for idx in 1:cfg.replications
                path = joinpath(task_dir(base_dir, SimulationTask(g_type, n, idx)), "results_dict.jls")
                if !isfile(path)
                    continue
                end
                results_dict = begin
                    file = open(path, "r")
                    data = deserialize(file)
                    close(file)
                    data
                end
                ns = results_dict["ns"]
                bi = results_dict["burn_in"]

                push!(betas_nonlinear_mat, mean(results_dict["beta_nonlinear"][1, (bi+1):ns]))
                push!(betas_se_nonlinear_mat, std(results_dict["beta_nonlinear"][1, (bi+1):ns]))
                push!(betas_nonlinear_mat2, mean(results_dict["beta_nonlinear"][2, (bi+1):ns]))
                push!(betas_se_nonlinear_mat2, std(results_dict["beta_nonlinear"][2, (bi+1):ns]))
                push!(H_nonlinear, mean(results_dict["Hseq"]))
                push!(K_nonlinear, mean(results_dict["Kseq"]))

                cr1 = quantile(results_dict["beta_nonlinear"][1, (bi+1):ns], [0.025, 0.975])
                push!(cr1_nonlinear, cr1[1] <= 0.5 <= cr1[2])
                cr2 = quantile(results_dict["beta_nonlinear"][2, (bi+1):ns], [0.025, 0.975])
                push!(cr2_nonlinear, cr2[1] <= 0.5 <= cr2[2])

                push!(betas_coxph_mat, mean(results_dict["beta_coxph"][1, (bi+1):ns]))
                push!(betas_se_coxph_mat, std(results_dict["beta_coxph"][1, (bi+1):ns]))
                push!(betas_coxph_mat2, mean(results_dict["beta_coxph"][2, (bi+1):ns]))
                push!(betas_se_coxph_mat2, std(results_dict["beta_coxph"][2, (bi+1):ns]))
                push!(H_coxph, mean(results_dict["Hseq_coxph"]))

                cr1c = quantile(results_dict["beta_coxph"][1, (bi+1):ns], [0.025, 0.975])
                push!(cr1_coxph, cr1c[1] <= 0.5 <= cr1c[2])
                cr2c = quantile(results_dict["beta_coxph"][2, (bi+1):ns], [0.025, 0.975])
                push!(cr2_coxph, cr2c[1] <= 0.5 <= cr2c[2])

                push!(df_IBS_summary, (n, g_type, "Nonlinear", results_dict["IBS_nonlinear"]))
                push!(df_IBS_summary, (n, g_type, "CoxPH", results_dict["IBS_coxph"]))
            end

            if !isempty(betas_nonlinear_mat)
                push!(df_summary, (
                    g_type, n, "Nonlinear",
                    mean(betas_nonlinear_mat) - 0.5,
                    mean(betas_se_nonlinear_mat),
                    std(betas_nonlinear_mat),
                    mean(cr1_nonlinear),
                    mean((betas_nonlinear_mat .- 0.5) .^ 2),
                    mean(betas_nonlinear_mat2) - 0.5,
                    mean(betas_se_nonlinear_mat2),
                    std(betas_nonlinear_mat2),
                    mean(cr2_nonlinear),
                    mean((betas_nonlinear_mat2 .- 0.5) .^ 2),
                    mean(H_nonlinear),
                    mean(K_nonlinear),
                ))

                push!(df_summary, (
                    g_type, n, "CoxPH",
                    mean(betas_coxph_mat) - 0.5,
                    mean(betas_se_coxph_mat),
                    std(betas_coxph_mat),
                    mean(cr1_coxph),
                    mean((betas_coxph_mat .- 0.5) .^ 2),
                    mean(betas_coxph_mat2) - 0.5,
                    mean(betas_se_coxph_mat2),
                    std(betas_coxph_mat2),
                    mean(cr2_coxph),
                    mean((betas_coxph_mat2 .- 0.5) .^ 2),
                    mean(H_coxph),
                    0.0,
                ))
            end
        end
    end

    CSV.write(joinpath(base_dir, "simu_summary.csv"), df_summary)
    CSV.write(joinpath(base_dir, "df_IBS.csv"), df_IBS_summary)
end

# CLI entry point.
function main()
    mode, reset, workers = parse_args()
    cfg = default_simulation_config(mode; n_workers=workers)
    base_dir = joinpath(RESULTS_DIR, "simulation", String(cfg.mode))

    if reset && isdir(base_dir)
        rm(base_dir; recursive=true, force=true)
    end
    ensure_results_dir(base_dir)

    tasks = SimulationTask[]
    for g in cfg.g_types
        for n in cfg.n_values
            for idx in 1:cfg.replications
                push!(tasks, SimulationTask(g, n, idx))
            end
        end
    end

    if isempty(tasks)
        println("No simulation tasks defined.")
        return
    end

    n_active_threads = min(cfg.n_workers, nthreads())
    println("Running simulations with $n_active_threads threads (set JULIA_NUM_THREADS to change).")
    prog = Progress(length(tasks); desc="Simulations")
    lock = ReentrantLock()
    limit = max(1, min(cfg.n_workers, nthreads()))
    sem = Base.Semaphore(limit)

    @sync begin
        for task in tasks
            Threads.@spawn begin
                Base.acquire(sem)
                try
                    run_single_task(task, cfg, base_dir)
                    lock() do
                        next!(prog)
                    end
                finally
                    Base.release(sem)
                end
            end
        end
    end

    summarize_tasks(cfg, base_dir)
    println("Simulation finished. Results saved to $(base_dir)")
end

main()
