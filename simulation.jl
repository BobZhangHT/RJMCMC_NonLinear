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
using Plots
using LaTeXStrings
using SpecialFunctions  # Required by model.jl for Dirichlet prior

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "model.jl"))

using .Config
using .RJMCMCModel

struct SimulationTask
    g_type::String  # true g(z) shape
    n::Int          # sample size
    idx::Int        # replication id
end

const METHOD_ALIASES = Dict{String, Symbol}(
    "all" => :all,
    "nonlinear1" => :nonlinear1,
    "nonlinear" => :nonlinear1,
    "nl1" => :nonlinear1,
    "nonlinear2" => :nonlinear2,
    "nonlinear_dirichlet" => :nonlinear2,
    "dirichlet" => :nonlinear2,
    "nl2" => :nonlinear2,
    "coxph" => :coxph,
    "cox" => :coxph,
)

const METHOD_KEYS = Dict{Symbol, Vector{String}}(
    :nonlinear1 => ["beta_nonlinear", "IBS_nonlinear", "Lambda_nonlinear", "g_nonlinear", "Hseq", "Kseq"],
    :nonlinear2 => ["beta_nonlinear_dirichlet", "IBS_nonlinear_dirichlet", "Lambda_nonlinear_dirichlet", "g_nonlinear_dirichlet",
        "Hseq_dirichlet", "Kseq_dirichlet", "alpha_tau_dirichlet", "alpha_zeta_dirichlet"],
    :coxph => ["beta_coxph", "IBS_coxph", "Lambda_coxph", "Hseq_coxph"],
)

function parse_method_list(arg_value::AbstractString)
    raw = lowercase(strip(arg_value))
    if isempty(raw) || raw == "all"
        return Set([:nonlinear1, :nonlinear2, :coxph])
    end

    parts = split(raw, ",")
    methods = Set{Symbol}()
    for p in parts
        key = lowercase(strip(p))
        if !haskey(METHOD_ALIASES, key) || METHOD_ALIASES[key] == :all
            error("Unknown method '$p'. Use --methods=nonlinear1,nonlinear2,coxph (or --methods=all).")
        end
        push!(methods, METHOD_ALIASES[key])
    end
    return methods
end

function method_done(results_dict::Dict, method::Symbol)
    keys = METHOD_KEYS[method]
    for k in keys
        if !haskey(results_dict, k)
            return false
        end
    end
    return true
end

# Parse command-line flags (--demo/--full/--reset/--replot/--plot-only/--workers=N/--methods=.../--rerun-methods=...).
function parse_args()
    mode = Config.full
    reset = false
    replot = false
    plot_only = false
    # Default to use all available CPU threads
    workers = nthreads()
    methods_to_run = Set([:nonlinear1, :nonlinear2, :coxph])
    rerun_methods = Set{Symbol}()

    for arg in ARGS
        if arg == "--demo"
            mode = Config.demo
        elseif arg == "--full"
            mode = Config.full
        elseif arg == "--reset"
            reset = true
        elseif arg == "--replot"
            replot = true
        elseif arg == "--plot-only"
            plot_only = true
        elseif startswith(arg, "--workers=")
            workers = parse(Int, split(arg, "=")[2])
        elseif arg == "--only-nonlinear2" || arg == "--only-nl2"
            methods_to_run = Set([:nonlinear2])
        elseif startswith(arg, "--methods=")
            methods_to_run = parse_method_list(split(arg, "=")[2])
        elseif startswith(arg, "--rerun-methods=")
            rerun_methods = parse_method_list(split(arg, "=")[2])
        elseif startswith(arg, "--force-methods=")
            rerun_methods = parse_method_list(split(arg, "=")[2])
        end
    end

    return mode, reset, replot, plot_only, workers, methods_to_run, rerun_methods
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
        elseif g_type == "linear"
            z -> 0.3 .* z
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

# True g(z) helper for plotting.
function g_true_values(g_type::String, z::AbstractVector{<:Real})
    if g_type == "sin"
        return sin.(z)
    elseif g_type == "quad"
        return -z .* (z .- 2pi) ./ 5
    elseif g_type == "linear"
        return 0.3 .* z
    elseif g_type == "U-quad"
        return z .* (z .- 2pi) ./ 5
    else
        return 1 ./(1 .+ exp.(2 .* (pi .- z)))
    end
end

baseline_cumhaz(t::AbstractVector{<:Real}, a::Real, b::Real) = a .* (t .^ b)

function ensure_results_dir(base_dir::String)
    if !isdir(base_dir)
        mkpath(base_dir)
    end
end

function task_dir(base_dir::String, task::SimulationTask)
    joinpath(base_dir, "g=$(task.g_type)", "n=$(task.n)", "rep=$(task.idx)")
end

function run_single_task(task::SimulationTask, cfg::SimulationConfig, base_dir::String, methods_to_run::Set{Symbol}, rerun_methods::Set{Symbol})
    println("[Task rep=$(task.idx)] Starting task execution on thread $(Threads.threadid())")
    
    # Performance timing
    timings = Dict{String, Float64}()
    t_start = time()
    t_last = t_start
    
    dir = task_dir(base_dir, task)
    results_path = joinpath(dir, "results_dict.jls")
    existing = Dict{Any, Any}()
    
    # Auto-resume: Load existing results if available (when not using --reset)
    if isfile(results_path)
        try
            t1 = time()
            open(results_path, "r") do io
                existing = deserialize(io)
            end
            timings["read_existing"] = time() - t1
            println("[Task rep=$(task.idx)] Loaded existing results from $(results_path)")
        catch err
            @warn "Failed to read existing results file $(results_path); will recompute selected methods." exception=(err, catch_backtrace())
            existing = Dict{Any, Any}()
        end
    else
        println("[Task rep=$(task.idx)] No existing results found. Starting fresh computation.")
    end
    
    t_last = time()
    timings["init"] = t_last - t_start

    mkpath(dir)
    train_path = joinpath(dir, "dat_train.csv")
    test_path = joinpath(dir, "dat_test.csv")

    # Generate data only if files don't exist, otherwise read from cache
    t1 = time()
    if !isfile(train_path) || !isfile(test_path)
        t_data = time()
        df_train = data_gen(task.idx; g_type=task.g_type, n=task.n, hazard_a=cfg.hazard_a, hazard_b=cfg.hazard_b, z_min=cfg.z_min, z_max=cfg.z_max)
        df_test = data_gen(task.idx + cfg.data_seed; g_type=task.g_type, n=cfg.n_test, hazard_a=cfg.hazard_a, hazard_b=cfg.hazard_b, z_min=cfg.z_min, z_max=cfg.z_max)
        timings["data_gen"] = time() - t_data
        
        t_write = time()
        CSV.write(train_path, df_train)
        CSV.write(test_path, df_test)
        timings["data_write"] = time() - t_write
        
        # Extract data directly from generated DataFrames to avoid re-reading
        Y_train, Delta_train = df_train.Y, df_train.Delta
        X_train = hcat(Float64.(df_train.X1), Float64.(df_train.X2))
        Z_train = df_train.Z
        Y_test, Delta_test = df_test.Y, df_test.Delta
        X_test = hcat(Float64.(df_test.X1), Float64.(df_test.X2))
        Z_test = df_test.Z
    else
        # Only read CSV if files already exist
        t_read = time()
        df_train = CSV.read(train_path, DataFrame)
        df_test = CSV.read(test_path, DataFrame)
        timings["data_read"] = time() - t_read
        
        Y_train, Delta_train = df_train.Y, df_train.Delta
        X_train = hcat(Float64.(df_train.X1), Float64.(df_train.X2))
        Z_train = df_train.Z
        Y_test, Delta_test = df_test.Y, df_test.Delta
        X_test = hcat(Float64.(df_test.X1), Float64.(df_test.X2))
        Z_test = df_test.Z
    end
    timings["data_io"] = time() - t1

    # Data-driven knot bounds used by the updated model (for reporting/traceability)
    obs_time = (Y_train .* Delta_train)
    obs_time = obs_time[obs_time .> 0]
    if !isempty(obs_time)
        tau_min = quantile(obs_time, 0.05)
        tau_max = quantile(obs_time, 0.95)
    else
        tau_min = 0.0
        tau_max = maximum(Y_train)
    end
    zeta_min = quantile(Z_train, 0.05)
    zeta_max = quantile(Z_train, 0.95)


    # Use known support for Z to avoid boundary bias/kinks when plotting on full grid.
    a, b = cfg.z_min, cfg.z_max

    parallel_run = cfg.n_workers > 1 && nthreads() > 1
    # Always disable MCMC progress bar, only show task-level progress
    show_mcmc_progress = false

    # Auto-resume logic: Check which methods need to be run
    # If method is in rerun_methods, force recomputation
    # Otherwise, check if method is already done in existing results
    need_nl1 = (:nonlinear1 in methods_to_run) && (:nonlinear1 in rerun_methods || !method_done(existing, :nonlinear1))
    need_nl2 = (:nonlinear2 in methods_to_run) && (:nonlinear2 in rerun_methods || !method_done(existing, :nonlinear2))
    need_cox = (:coxph in methods_to_run) && (:coxph in rerun_methods || !method_done(existing, :coxph))

    if !(need_nl1 || need_nl2 || need_cox)
        println("[Task rep=$(task.idx)] All requested methods already completed. Skipping task.")
        return :skipped
    end
    
    # Print which methods will be run/skipped for better visibility
    methods_status = String[]
    if :nonlinear1 in methods_to_run
        if need_nl1
            push!(methods_status, "NonLinear1: will run")
        else
            push!(methods_status, "NonLinear1: already done, skipping")
        end
    end
    if :nonlinear2 in methods_to_run
        if need_nl2
            push!(methods_status, "NonLinear2: will run")
        else
            push!(methods_status, "NonLinear2: already done, skipping")
        end
    end
    if :coxph in methods_to_run
        if need_cox
            push!(methods_status, "CoxPH: will run")
        else
            push!(methods_status, "CoxPH: already done, skipping")
        end
    end
    println("[Task rep=$(task.idx)] Methods status: $(join(methods_status, ", "))")

    ns = cfg.ns
    bi = cfg.burn_in
    t_grid = cfg.t_grid
    z_grid = cfg.z_grid

    g_tmp = zeros(length(z_grid))
    g_knots = Vector{Float64}(undef, DEFAULT_KMAX + 2)
    Lambda_tmp = zeros(length(t_grid))
    h_knots = Vector{Float64}(undef, DEFAULT_HMAX + 2)
    h_cumhaz = Vector{Float64}(undef, DEFAULT_HMAX + 2)

    results_dict = existing

    if need_nl1
        println("[Task rep=$(task.idx)] Starting NonLinear1 MCMC...")
        t_nl1_start = time()
        # Use unique seed per task to avoid RNG conflicts in parallel execution
        mcmc_seed_nl1 = cfg.mcmc_seed + task.idx * 10000
        t_mcmc = time()
        results_nonlinear = RJMCMC_Nonlinear(X_train, Z_train, Delta_train, Y_train, a, b, mcmc_seed_nl1;
            ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX, Kmax=DEFAULT_KMAX, show_progress=show_mcmc_progress)
        timings["nl1_mcmc"] = time() - t_mcmc
        println("[Task rep=$(task.idx)] NonLinear1 MCMC completed in $(round(timings["nl1_mcmc"], digits=2))s")

        K_all, zetas_all, xis_all = results_nonlinear["K"], results_nonlinear["zetas"], results_nonlinear["xis"]
        H_all, taus_all, gammas_all = results_nonlinear["H"], results_nonlinear["taus"], results_nonlinear["gammas"]
        Tmax = results_nonlinear["Tmax"]

        t_g = time()
        g_pos = zeros(length(z_grid))
        for i in (bi + 1):ns
            K = Int(K_all[i])
            zetas = zetas_all[1:K, i]
            xis = xis_all[1:(K + 2), i]
            RJMCMCModel.g_fun_est!(g_tmp, z_grid, a, b, zetas, xis, g_knots)
            g_pos .+= g_tmp
        end
        g_pos ./= (ns - bi)
        timings["nl1_g_est"] = time() - t_g

        t_lambda = time()
        Lambda_nonlinear = zeros(length(t_grid))
        for i in (bi + 1):ns
            H = Int(H_all[i])
            taus = taus_all[1:H, i]
            gammas = gammas_all[1:(H + 2), i]
            RJMCMCModel.Lambda_fun_est!(Lambda_tmp, t_grid, Tmax, taus, gammas, h_knots, h_cumhaz)
            Lambda_nonlinear .+= Lambda_tmp
        end
        Lambda_nonlinear ./= (ns - bi)
        timings["nl1_lambda_est"] = time() - t_lambda

        println("[Task rep=$(task.idx)] Computing NonLinear1 IBS...")
        t_ibs = time()
        # Use reduced n_int for faster IBS computation (can be adjusted for accuracy vs speed tradeoff)
        # Default n_int=200 is very slow; using 50 provides good balance with acceptable accuracy
        # The IBS function will automatically use sparse MCMC sampling when n_samples > 2000
        IBS_nonlinear = IBS(Y_train, Delta_train, X_train, Z_train, Y_test, Delta_test, X_test, Z_test, results_nonlinear, "nonlinear", 50, cfg.mcmc_seed; show_progress=false)
        timings["nl1_ibs"] = time() - t_ibs
        println("[Task rep=$(task.idx)] NonLinear1 IBS completed in $(round(timings["nl1_ibs"], digits=2))s")
        
        timings["nl1_total"] = time() - t_nl1_start

        results_dict["beta_nonlinear"] = results_nonlinear["betas"]
        results_dict["IBS_nonlinear"] = IBS_nonlinear
        results_dict["Lambda_nonlinear"] = Lambda_nonlinear
        results_dict["g_nonlinear"] = g_pos
        results_dict["Hseq"] = results_nonlinear["H"][bi+1:ns]
        results_dict["Kseq"] = results_nonlinear["K"][bi+1:ns]
        results_dict["Tmax"] = Tmax
    end

    if need_nl2
        println("[Task rep=$(task.idx)] Starting NonLinear2 MCMC...")
        t_nl2_start = time()
        # Use unique seed per task to avoid RNG conflicts in parallel execution
        mcmc_seed_nl2 = cfg.mcmc_seed + 1000 + task.idx * 10000
        t_mcmc = time()
        results_nonlinear_dirichlet = RJMCMC_Nonlinear_Dirichlet(X_train, Z_train, Delta_train, Y_train, a, b, mcmc_seed_nl2;
            ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX, Kmax=DEFAULT_KMAX, 
            a_tau=1.0, b_tau=1.0, a_zeta=1.0, b_zeta=1.0, show_progress=show_mcmc_progress)
        timings["nl2_mcmc"] = time() - t_mcmc
        println("[Task rep=$(task.idx)] NonLinear2 MCMC completed in $(round(timings["nl2_mcmc"], digits=2))s")

        K_all_dir, zetas_all_dir, xis_all_dir = results_nonlinear_dirichlet["K"], results_nonlinear_dirichlet["zetas"], results_nonlinear_dirichlet["xis"]
        H_all_dir, taus_all_dir, gammas_all_dir = results_nonlinear_dirichlet["H"], results_nonlinear_dirichlet["taus"], results_nonlinear_dirichlet["gammas"]
        Tmax_dir = results_nonlinear_dirichlet["Tmax"]

        t_g = time()
        g_pos_dirichlet = zeros(length(z_grid))
        for i in (bi + 1):ns
            K = Int(K_all_dir[i])
            zetas = zetas_all_dir[1:K, i]
            xis = xis_all_dir[1:(K + 2), i]
            RJMCMCModel.g_fun_est!(g_tmp, z_grid, a, b, zetas, xis, g_knots)
            g_pos_dirichlet .+= g_tmp
        end
        g_pos_dirichlet ./= (ns - bi)
        timings["nl2_g_est"] = time() - t_g

        t_lambda = time()
        Lambda_nonlinear_dirichlet = zeros(length(t_grid))
        for i in (bi + 1):ns
            H = Int(H_all_dir[i])
            taus = taus_all_dir[1:H, i]
            gammas = gammas_all_dir[1:(H + 2), i]
            RJMCMCModel.Lambda_fun_est!(Lambda_tmp, t_grid, Tmax_dir, taus, gammas, h_knots, h_cumhaz)
            Lambda_nonlinear_dirichlet .+= Lambda_tmp
        end
        Lambda_nonlinear_dirichlet ./= (ns - bi)
        timings["nl2_lambda_est"] = time() - t_lambda

        println("[Task rep=$(task.idx)] Computing NonLinear2 IBS...")
        t_ibs = time()
        # Use reduced n_int for faster IBS computation
        # The IBS function will automatically use sparse MCMC sampling when n_samples > 2000
        IBS_nonlinear_dirichlet = IBS(Y_train, Delta_train, X_train, Z_train, Y_test, Delta_test, X_test, Z_test, results_nonlinear_dirichlet, "nonlinear", 50, cfg.mcmc_seed + 1000; show_progress=false)
        timings["nl2_ibs"] = time() - t_ibs
        println("[Task rep=$(task.idx)] NonLinear2 IBS completed in $(round(timings["nl2_ibs"], digits=2))s")
        
        timings["nl2_total"] = time() - t_nl2_start

        results_dict["beta_nonlinear_dirichlet"] = results_nonlinear_dirichlet["betas"]
        results_dict["IBS_nonlinear_dirichlet"] = IBS_nonlinear_dirichlet
        results_dict["Lambda_nonlinear_dirichlet"] = Lambda_nonlinear_dirichlet
        results_dict["g_nonlinear_dirichlet"] = g_pos_dirichlet
        results_dict["Hseq_dirichlet"] = results_nonlinear_dirichlet["H"][bi+1:ns]
        results_dict["Kseq_dirichlet"] = results_nonlinear_dirichlet["K"][bi+1:ns]
        results_dict["alpha_tau_dirichlet"] = results_nonlinear_dirichlet["alpha_tau"][bi+1:ns]
        results_dict["alpha_zeta_dirichlet"] = results_nonlinear_dirichlet["alpha_zeta"][bi+1:ns]
        if !haskey(results_dict, "Tmax")
            results_dict["Tmax"] = Tmax_dir
        end
    end

    if need_cox
        println("[Task rep=$(task.idx)] Starting CoxPH MCMC...")
        t_cox_start = time()
        # Use unique seed per task to avoid RNG conflicts in parallel execution
        mcmc_seed_cox = cfg.mcmc_seed + task.idx * 10000
        t_mcmc = time()
        results_coxph = RJMCMC_CoxPH(X_train, Z_train, Delta_train, Y_train, mcmc_seed_cox;
            ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX, show_progress=show_mcmc_progress)
        timings["cox_mcmc"] = time() - t_mcmc
        println("[Task rep=$(task.idx)] CoxPH MCMC completed in $(round(timings["cox_mcmc"], digits=2))s")

        H_cox_all, taus_cox_all, gammas_cox_all = results_coxph["H"], results_coxph["taus"], results_coxph["gammas"]
        Tmax_cox = get(results_coxph, "Tmax", results_coxph["tau"])

        t_lambda = time()
        Lambda_coxph = zeros(length(t_grid))
        for i in (bi + 1):ns
            H = Int(H_cox_all[i])
            taus = taus_cox_all[1:H, i]
            gammas = gammas_cox_all[1:(H + 2), i]
            RJMCMCModel.Lambda_fun_est!(Lambda_tmp, t_grid, Tmax_cox, taus, gammas, h_knots, h_cumhaz)
            Lambda_coxph .+= Lambda_tmp
        end
        Lambda_coxph ./= (ns - bi)
        timings["cox_lambda_est"] = time() - t_lambda

        println("[Task rep=$(task.idx)] Computing CoxPH IBS...")
        t_ibs = time()
        # Use reduced n_int for faster IBS computation
        # The IBS function will automatically use sparse MCMC sampling when n_samples > 2000
        IBS_coxph = IBS(Y_train, Delta_train, X_train, Z_train, Y_test, Delta_test, X_test, Z_test, results_coxph, "coxph", 50, cfg.mcmc_seed; show_progress=false)
        timings["cox_ibs"] = time() - t_ibs
        println("[Task rep=$(task.idx)] CoxPH IBS completed in $(round(timings["cox_ibs"], digits=2))s")
        
        timings["cox_total"] = time() - t_cox_start

        results_dict["beta_coxph"] = results_coxph["betas"]
        results_dict["IBS_coxph"] = IBS_coxph
        results_dict["Lambda_coxph"] = Lambda_coxph
        results_dict["Hseq_coxph"] = results_coxph["H"][bi+1:ns]
        if !haskey(results_dict, "Tmax")
            results_dict["Tmax"] = Tmax_cox
        end
    end

    results_dict["ns"] = ns
    results_dict["burn_in"] = bi
    results_dict["mode"] = string(cfg.mode)
    results_dict["tau_min"] = tau_min
    results_dict["tau_max"] = tau_max
    results_dict["zeta_min"] = zeta_min
    results_dict["zeta_max"] = zeta_max

    # Optimized file writing: use atomic write with retry for Windows compatibility
    # Use unique temp filename per task to avoid conflicts when tasks run in parallel
    t_serialize = time()
    task_id = hash((task.g_type, task.n, task.idx))
    tmp_path = results_path * ".tmp_$(task_id)"
    try
        t_serialize_start = time()
        open(tmp_path, "w") do file
            serialize(file, results_dict)
        end
        timings["serialize"] = time() - t_serialize_start
        
        t_file_move = time()
        # Windows file locking workaround: use retry with exponential backoff
        written = false
        max_retries = 5
        for retry in 1:max_retries
            try
                # Try atomic move first (fastest)
                mv(tmp_path, results_path; force=true)
                written = true
                break
            catch e
                if retry < max_retries
                    # Wait with exponential backoff (Windows antivirus/file indexing can lock files)
                    sleep(0.1 * retry)
                    # Try copy-and-remove as fallback
                    try
                        cp(tmp_path, results_path; force=true)
                        # Try to remove temp file, but don't fail if it's locked
                        try
                            rm(tmp_path; force=true)
                        catch
                            # File might be locked, will be cleaned up later
                        end
                        written = true
                        break
                    catch
                        # Continue to next retry
                    end
                else
                    # Last attempt: try copy without removing temp file
                    try
                        cp(tmp_path, results_path; force=true)
                        written = true
                        break
                    catch
                        # If all else fails, rethrow the original error
                        rethrow(e)
                    end
                end
            end
        end
        if !written
            error("Failed to write results file after $(max_retries) attempts")
        end
        timings["file_write"] = time() - t_file_move
    catch
        # Clean up temp file if something went wrong
        if isfile(tmp_path)
            try rm(tmp_path; force=true) catch end
        end
        rethrow()
    end
    timings["save_total"] = time() - t_serialize

    timings["total"] = time() - t_start
    
    # Store timings in a thread-safe way (using lock for printing)
    task_label = "g=$(task.g_type),n=$(task.n),rep=$(task.idx)"
    
    # Print timing summary (thread-safe with lock)
    timing_str = "\n[Timing] $task_label (thread $(Threads.threadid())):\n"
    total_time = timings["total"]
    # Sort by value (time) in descending order
    sorted_timings = sort(collect(pairs(timings)), by=x->x.second, rev=true)
    for (key, val) in sorted_timings
        pct = total_time > 0 ? round(100*val/total_time, digits=1) : 0.0
        timing_str *= "  $(rpad(key, 25)): $(rpad(round(val, digits=3), 8))s ($(pct)%)\n"
    end
    println(timing_str)

    return :updated
end

# Aggregate posterior summaries and IBS metrics across completed replications.
function summarize_tasks(cfg::SimulationConfig, base_dir::String)
    plots_dir = joinpath(base_dir, "plots")
    mkpath(plots_dir)
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
            # NonLinear1 (Default Prior) storage
            betas_nonlinear_mat = Float64[]
            betas_se_nonlinear_mat = Float64[]
            betas_nonlinear_mat2 = Float64[]
            betas_se_nonlinear_mat2 = Float64[]
            H_nonlinear = Float64[]
            K_nonlinear = Float64[]
            cr1_nonlinear = Float64[]
            cr2_nonlinear = Float64[]

            # NonLinear2 (Dirichlet-Gamma Prior) storage
            betas_nonlinear_dir_mat = Float64[]
            betas_se_nonlinear_dir_mat = Float64[]
            betas_nonlinear_dir_mat2 = Float64[]
            betas_se_nonlinear_dir_mat2 = Float64[]
            H_nonlinear_dir = Float64[]
            K_nonlinear_dir = Float64[]
            cr1_nonlinear_dir = Float64[]
            cr2_nonlinear_dir = Float64[]
            alpha_tau_dir = Float64[]
            alpha_zeta_dir = Float64[]

            # CoxPH storage
            betas_coxph_mat = Float64[]
            betas_se_coxph_mat = Float64[]
            betas_coxph_mat2 = Float64[]
            betas_se_coxph_mat2 = Float64[]
            H_coxph = Float64[]
            cr1_coxph = Float64[]
            cr2_coxph = Float64[]

            g_samples = Vector{Vector{Float64}}()
            g_samples_dir = Vector{Vector{Float64}}()
            lambda_non_samples = Vector{Vector{Float64}}()
            lambda_non_dir_samples = Vector{Vector{Float64}}()
            lambda_cox_samples = Vector{Vector{Float64}}()

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

                # NonLinear1 (Default Prior)
                if haskey(results_dict, "beta_nonlinear")
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

                    if haskey(results_dict, "g_nonlinear")
                        push!(g_samples, results_dict["g_nonlinear"])
                    end
                    if haskey(results_dict, "Lambda_nonlinear")
                        push!(lambda_non_samples, results_dict["Lambda_nonlinear"])
                    end
                end

                # NonLinear2 (Dirichlet-Gamma Prior)
                if haskey(results_dict, "beta_nonlinear_dirichlet")
                    push!(betas_nonlinear_dir_mat, mean(results_dict["beta_nonlinear_dirichlet"][1, (bi+1):ns]))
                    push!(betas_se_nonlinear_dir_mat, std(results_dict["beta_nonlinear_dirichlet"][1, (bi+1):ns]))
                    push!(betas_nonlinear_dir_mat2, mean(results_dict["beta_nonlinear_dirichlet"][2, (bi+1):ns]))
                    push!(betas_se_nonlinear_dir_mat2, std(results_dict["beta_nonlinear_dirichlet"][2, (bi+1):ns]))
                    push!(H_nonlinear_dir, mean(results_dict["Hseq_dirichlet"]))
                    push!(K_nonlinear_dir, mean(results_dict["Kseq_dirichlet"]))
                    push!(alpha_tau_dir, mean(results_dict["alpha_tau_dirichlet"]))
                    push!(alpha_zeta_dir, mean(results_dict["alpha_zeta_dirichlet"]))

                    cr1_dir = quantile(results_dict["beta_nonlinear_dirichlet"][1, (bi+1):ns], [0.025, 0.975])
                    push!(cr1_nonlinear_dir, cr1_dir[1] <= 0.5 <= cr1_dir[2])
                    cr2_dir = quantile(results_dict["beta_nonlinear_dirichlet"][2, (bi+1):ns], [0.025, 0.975])
                    push!(cr2_nonlinear_dir, cr2_dir[1] <= 0.5 <= cr2_dir[2])

                    push!(g_samples_dir, results_dict["g_nonlinear_dirichlet"])
                    push!(lambda_non_dir_samples, results_dict["Lambda_nonlinear_dirichlet"])
                end

                # CoxPH
                if haskey(results_dict, "beta_coxph")
                    push!(betas_coxph_mat, mean(results_dict["beta_coxph"][1, (bi+1):ns]))
                    push!(betas_se_coxph_mat, std(results_dict["beta_coxph"][1, (bi+1):ns]))
                    push!(betas_coxph_mat2, mean(results_dict["beta_coxph"][2, (bi+1):ns]))
                    push!(betas_se_coxph_mat2, std(results_dict["beta_coxph"][2, (bi+1):ns]))
                    push!(H_coxph, mean(results_dict["Hseq_coxph"]))
                    if haskey(results_dict, "Lambda_coxph")
                        push!(lambda_cox_samples, results_dict["Lambda_coxph"])
                    end

                    cr1c = quantile(results_dict["beta_coxph"][1, (bi+1):ns], [0.025, 0.975])
                    push!(cr1_coxph, cr1c[1] <= 0.5 <= cr1c[2])
                    cr2c = quantile(results_dict["beta_coxph"][2, (bi+1):ns], [0.025, 0.975])
                    push!(cr2_coxph, cr2c[1] <= 0.5 <= cr2c[2])
                end

                # IBS entries for all methods
                if haskey(results_dict, "IBS_nonlinear")
                    push!(df_IBS_summary, (n, g_type, "NonLinear1", results_dict["IBS_nonlinear"]))
                end
                if haskey(results_dict, "IBS_coxph")
                    push!(df_IBS_summary, (n, g_type, "CoxPH", results_dict["IBS_coxph"]))
                end
                if haskey(results_dict, "IBS_nonlinear_dirichlet")
                    push!(df_IBS_summary, (n, g_type, "NonLinear2", results_dict["IBS_nonlinear_dirichlet"]))
                end
            end

            if !isempty(betas_nonlinear_mat) || !isempty(betas_nonlinear_dir_mat) || !isempty(betas_coxph_mat)
                # NonLinear1 (Default Prior) summary
                if !isempty(betas_nonlinear_mat)
                    push!(df_summary, (
                        g_type, n, "NonLinear1",
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
                end

                # NonLinear2 (Dirichlet-Gamma Prior) summary
                if !isempty(betas_nonlinear_dir_mat)
                    push!(df_summary, (
                        g_type, n, "NonLinear2",
                        mean(betas_nonlinear_dir_mat) - 0.5,
                        mean(betas_se_nonlinear_dir_mat),
                        std(betas_nonlinear_dir_mat),
                        mean(cr1_nonlinear_dir),
                        mean((betas_nonlinear_dir_mat .- 0.5) .^ 2),
                        mean(betas_nonlinear_dir_mat2) - 0.5,
                        mean(betas_se_nonlinear_dir_mat2),
                        std(betas_nonlinear_dir_mat2),
                        mean(cr2_nonlinear_dir),
                        mean((betas_nonlinear_dir_mat2 .- 0.5) .^ 2),
                        mean(H_nonlinear_dir),
                        mean(K_nonlinear_dir),
                    ))
                end

                # CoxPH summary
                if !isempty(betas_coxph_mat)
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

                # --- Plotting (PDF) ---
                z_grid = cfg.z_grid
                t_grid = cfg.t_grid
                g_truth = g_true_values(g_type, z_grid)
                baseline = baseline_cumhaz(t_grid, cfg.hazard_a, cfg.hazard_b)

                # g(z) plot with both NonLinear methods
                if !isempty(g_samples)
                    g_mat = hcat(g_samples...)
                    g_mean = vec(mean(g_mat, dims=2))

                    # Legend position: topleft for linear, topright for others
                    legend_pos_g = (g_type == "linear") ? :topleft : :topright
                    
                    plt_g = plot(z_grid, g_truth; lw=2, color=:black, label="True", legend=legend_pos_g)
                    plot!(plt_g, z_grid, g_mean; lw=2, color=:red, linestyle=:dot, label="NonLinear1")
                    
                    if !isempty(g_samples_dir)
                        g_mat_dir = hcat(g_samples_dir...)
                        g_mean_dir = vec(mean(g_mat_dir, dims=2))
                        plot!(plt_g, z_grid, g_mean_dir; lw=2, color=:green, linestyle=:dash, label="NonLinear2")
                    end
                    
                    xlabel!(plt_g, "z")
                    ylabel!(plt_g, "g(z)")
                    title!(plt_g, "n = $(n)")
                    savefig(plt_g, joinpath(plots_dir, "g_$(g_type)_n$(n).pdf"))
                end

                # Lambda(t) plot with both NonLinear methods
                if !isempty(lambda_non_samples)
                    lam_non_mat = hcat(lambda_non_samples...)
                    lam_non_mean = vec(mean(lam_non_mat, dims=2))

                    plt_lam = plot(t_grid, baseline; lw=2, color=:black, label="True", legend=:topleft)
                    plot!(plt_lam, t_grid, lam_non_mean; lw=2, color=:red, linestyle=:dot, label="NonLinear1")
                    
                    if !isempty(lambda_non_dir_samples)
                        lam_non_dir_mat = hcat(lambda_non_dir_samples...)
                        lam_non_dir_mean = vec(mean(lam_non_dir_mat, dims=2))
                        plot!(plt_lam, t_grid, lam_non_dir_mean; lw=2, color=:green, linestyle=:dash, label="NonLinear2")
                    end
                    
                    xlabel!(plt_lam, "t")
                    ylabel!(plt_lam, "Lambda(t)")
                    title!(plt_lam, "n = $(n)")
                    savefig(plt_lam, joinpath(plots_dir, "lambda_$(g_type)_n$(n).pdf"))
                end
            end
        end
    end

    CSV.write(joinpath(base_dir, "simu_summary.csv"), df_summary)
    CSV.write(joinpath(base_dir, "df_IBS.csv"), df_IBS_summary)
end

# -------------------------------------------------------------------------
# Manuscript-style plots (Figures 1-4) with CoxPH comparisons
# Compares: CoxPH, NonLinear1 (Default Prior), NonLinear2 (Dirichlet-Gamma Prior)
# -------------------------------------------------------------------------
function generate_manuscript_plots(cfg::SimulationConfig, base_dir::String)
    plot_dir = joinpath(base_dir, "plots_manuscript")
    mkpath(plot_dir)

    baseline = baseline_cumhaz(cfg.t_grid, cfg.hazard_a, cfg.hazard_b)
    ibs_rows = DataFrame(g_type=String[], n=Int[], Method=String[], IBS=Float64[])

    # Cache means per (g, n) for all methods
    lam_non1_means = Dict{Tuple{String,Int}, Vector{Float64}}()  # NonLinear1 (Default)
    lam_non2_means = Dict{Tuple{String,Int}, Vector{Float64}}()  # NonLinear2 (Dirichlet)
    lam_cox_means = Dict{Tuple{String,Int}, Union{Vector{Float64},Nothing}}()
    g_non1_means = Dict{Tuple{String,Int}, Vector{Float64}}()    # NonLinear1 (Default)
    g_non2_means = Dict{Tuple{String,Int}, Vector{Float64}}()    # NonLinear2 (Dirichlet)

    for g_type in cfg.g_types
        for n in cfg.n_values
            lam_non1_list = Vector{Vector{Float64}}()
            lam_non2_list = Vector{Vector{Float64}}()
            lam_cox_list = Vector{Vector{Float64}}()
            g_non1_list = Vector{Vector{Float64}}()
            g_non2_list = Vector{Vector{Float64}}()

            # Collect only existing replications (use max up to cfg.replications but skip missing)
            found_any = false
            for idx in 1:cfg.replications
                path = joinpath(task_dir(base_dir, SimulationTask(g_type, n, idx)), "results_dict.jls")
                if !isfile(path)
                    continue
                end
                found_any = true
                file = open(path, "r")
                data = deserialize(file)
                close(file)

                # NonLinear1 (Default Prior)
                if haskey(data, "Lambda_nonlinear")
                    push!(lam_non1_list, data["Lambda_nonlinear"])
                end
                if haskey(data, "g_nonlinear")
                    push!(g_non1_list, data["g_nonlinear"])
                end
                
                # NonLinear2 (Dirichlet-Gamma Prior)
                if haskey(data, "Lambda_nonlinear_dirichlet")
                    push!(lam_non2_list, data["Lambda_nonlinear_dirichlet"])
                end
                if haskey(data, "g_nonlinear_dirichlet")
                    push!(g_non2_list, data["g_nonlinear_dirichlet"])
                end
                
                # CoxPH
                if haskey(data, "Lambda_coxph")
                    push!(lam_cox_list, data["Lambda_coxph"])
                end
                
                # IBS for all methods
                if haskey(data, "IBS_nonlinear")
                    push!(ibs_rows, (g_type, n, "NonLinear1", data["IBS_nonlinear"]))
                end
                if haskey(data, "IBS_nonlinear_dirichlet")
                    push!(ibs_rows, (g_type, n, "NonLinear2", data["IBS_nonlinear_dirichlet"]))
                end
                if haskey(data, "IBS_coxph")
                    push!(ibs_rows, (g_type, n, "CoxPH", data["IBS_coxph"]))
                end
            end

            if !found_any
                @warn "No results found for g=$(g_type), n=$(n) in $(base_dir); plots may be empty. Rerun simulations for this setting."
                continue
            end

            # Store means for NonLinear1
            if !isempty(lam_non1_list)
                lam_non1_means[(g_type, n)] = vec(mean(hcat(lam_non1_list...), dims=2))
            end
            if !isempty(g_non1_list)
                g_non1_means[(g_type, n)] = vec(mean(hcat(g_non1_list...), dims=2))
            end
            
            # Store means for NonLinear2
            if !isempty(lam_non2_list)
                lam_non2_means[(g_type, n)] = vec(mean(hcat(lam_non2_list...), dims=2))
            end
            if !isempty(g_non2_list)
                g_non2_means[(g_type, n)] = vec(mean(hcat(g_non2_list...), dims=2))
            end
            
            # Store means for CoxPH
            if !isempty(lam_cox_list)
                lam_cox_means[(g_type, n)] = vec(mean(hcat(lam_cox_list...), dims=2))
            else
                lam_cox_means[(g_type, n)] = nothing
            end
        end
    end

    # Figures for each g_type
    for g_type in cfg.g_types
        # Collect available n values for this g_type
        available_n_lam = [n for n in cfg.n_values if haskey(lam_non1_means, (g_type, n)) || haskey(lam_non2_means, (g_type, n))]
        available_n_g = [n for n in cfg.n_values if haskey(g_non1_means, (g_type, n)) || haskey(g_non2_means, (g_type, n))]
        
        # Lambda panels - compare NonLinear1 and NonLinear2
        if !isempty(available_n_lam)
            n_lam_count = length(available_n_lam)
            lam_layout = (1, n_lam_count)
            p_lam = plot(layout=lam_layout, size=(600 * n_lam_count, 400), legend=true)
            for (i, n) in enumerate(available_n_lam)
                has_non1 = haskey(lam_non1_means, (g_type, n))
                if has_non1
                    lam_non1 = lam_non1_means[(g_type, n)]
                    # Handle NaN values
                    lam_non1_clean = [isnan(x) ? missing : x for x in lam_non1]
                    lam_non1_avg = [ismissing(x) ? 0.0 : x for x in lam_non1_clean]
                end
                
                # Check if NonLinear2 data exists
                has_non2 = haskey(lam_non2_means, (g_type, n))
                
                if has_non2
                    lam_non2 = lam_non2_means[(g_type, n)]
                    lam_non2_clean = [isnan(x) ? missing : x for x in lam_non2]
                    lam_non2_avg = [ismissing(x) ? 0.0 : x for x in lam_non2_clean]
                end
                
                if i == 1
                    plot!(p_lam[i], cfg.t_grid, baseline,
                          xlabel=L"$t$", ylabel=L"$\Lambda(t)$", margin=10Plots.mm,
                          label="True", color=:black, lw=2, title="n = $(n)", 
                          xlim=(0, 2.5), ylim=(0, 5),
                          tickfontsize=12,
                          xguidefontsize=14,
                          yguidefontsize=14,
                          legend=:topleft)
                    if has_non1
                        plot!(p_lam[i], cfg.t_grid, lam_non1_avg, label="NonLinear1", color=:red, lw=2, linestyle=:dot)
                    end
                    if has_non2
                        plot!(p_lam[i], cfg.t_grid, lam_non2_avg, label="NonLinear2", color=:green, lw=2, linestyle=:dash)
                    end
                else
                    plot!(p_lam[i], cfg.t_grid, baseline,
                          xlabel=L"$t$", ylabel=L"$\Lambda(t)$", margin=10Plots.mm,
                          label="", color=:black, lw=2, title="n = $(n)",
                          xlim=(0, 2.5), ylim=(0, 5),
                          tickfontsize=12,
                          xguidefontsize=14,
                          yguidefontsize=14)
                    if has_non1
                        plot!(p_lam[i], cfg.t_grid, lam_non1_avg, label="", color=:red, lw=2, linestyle=:dot)
                    end
                    if has_non2
                        plot!(p_lam[i], cfg.t_grid, lam_non2_avg, label="", color=:green, lw=2, linestyle=:dash)
                    end
                end
            end
            savefig(p_lam, joinpath(plot_dir, "Lambda_$(g_type).pdf"))
            println("Generated Lambda plot for g=$(g_type) with $(n_lam_count) sample size(s): $(available_n_lam)")
        else
            @warn "No Lambda data available for g=$(g_type); skipping plot."
        end

        # g(z) panels - compare NonLinear1 and NonLinear2
        if !isempty(available_n_g)
            n_g_count = length(available_n_g)
            g_layout = (1, n_g_count)
            p_g = plot(layout=g_layout, size=(600 * n_g_count, 400), legend=true)
            g_true = g_true_values(g_type, cfg.z_grid)
            for (i, n) in enumerate(available_n_g)
                has_non1 = haskey(g_non1_means, (g_type, n))
                if has_non1
                    g_non1 = g_non1_means[(g_type, n)]
                    # Handle NaN values
                    g_non1_clean = [isnan(x) ? missing : x for x in g_non1]
                    g_non1_avg = [ismissing(x) ? 0.0 : x for x in g_non1_clean]
                end
                
                # Check if NonLinear2 data exists
                has_non2 = haskey(g_non2_means, (g_type, n))
                
                if has_non2
                    g_non2 = g_non2_means[(g_type, n)]
                    g_non2_clean = [isnan(x) ? missing : x for x in g_non2]
                    g_non2_avg = [ismissing(x) ? 0.0 : x for x in g_non2_clean]
                end
                
                # Legend position: topleft for linear, topright for others
                legend_pos_g = (g_type == "linear") ? :topleft : :topright
                
                if i == 1
                    plot!(p_g[i], cfg.z_grid, g_true,
                          xlabel=L"$z$", ylabel=L"$g(z)$", margin=10Plots.mm,
                          label="True", color=:black, lw=2, title="n = $(n)",
                          tickfontsize=12,
                          xguidefontsize=14,
                          yguidefontsize=14,
                          legend=legend_pos_g)
                    if has_non1
                        plot!(p_g[i], cfg.z_grid, g_non1_avg, label="NonLinear1", color=:red, lw=2, linestyle=:dot)
                    end
                    if has_non2
                        plot!(p_g[i], cfg.z_grid, g_non2_avg, label="NonLinear2", color=:green, lw=2, linestyle=:dash)
                    end
                else
                    plot!(p_g[i], cfg.z_grid, g_true,
                          xlabel=L"$z$", ylabel=L"$g(z)$", margin=10Plots.mm,
                          label="", color=:black, lw=2, title="n = $(n)",
                          tickfontsize=12,
                          xguidefontsize=14,
                          yguidefontsize=14)
                    if has_non1
                        plot!(p_g[i], cfg.z_grid, g_non1_avg, label="", color=:red, lw=2, linestyle=:dot)
                    end
                    if has_non2
                        plot!(p_g[i], cfg.z_grid, g_non2_avg, label="", color=:green, lw=2, linestyle=:dash)
                    end
                end
            end
            savefig(p_g, joinpath(plot_dir, "g_$(g_type).pdf"))
            println("Generated g(z) plot for g=$(g_type) with $(n_g_count) sample size(s): $(available_n_g)")
        else
            @warn "No g(z) data available for g=$(g_type); skipping plot."
        end
    end

    # IBS boxplot using R script (Figure 4 style)
    # Use the df_IBS.csv file generated by summarize_tasks
    ibs_csv_path = joinpath(base_dir, "df_IBS.csv")
    ibs_pdf_path = joinpath(plot_dir, "IBS_boxplots.pdf")
    
    if isfile(ibs_csv_path)
        # Call R script to generate IBS boxplot
        r_script_path = joinpath(@__DIR__, "boxplot.R")
        if isfile(r_script_path)
            try
                # Ensure output directory exists
                mkpath(dirname(ibs_pdf_path))
                run(`Rscript $r_script_path $ibs_csv_path $ibs_pdf_path`)
                if isfile(ibs_pdf_path)
                    println("Generated IBS boxplot using R script: $(ibs_pdf_path)")
                else
                    @warn "R script completed but output file not found: $(ibs_pdf_path)"
                end
            catch e
                @warn "Failed to generate IBS boxplot using R script: $e"
                @warn "Make sure R and ggplot2 package are installed."
                @warn "You can install ggplot2 in R with: install.packages('ggplot2')"
            end
        else
            @warn "R script not found at $(r_script_path); skipping IBS boxplot generation."
        end
    else
        @warn "IBS CSV file not found at $(ibs_csv_path); skipping IBS boxplot generation."
        @warn "Run summarize_tasks first to generate df_IBS.csv"
    end
end

# Generate plots only (without running simulations)
function generate_plots_only(mode::SimulationMode=Config.demo)
    cfg = default_simulation_config(mode)
    base_dir = joinpath(RESULTS_DIR, "simulation", string(cfg.mode))
    
    if !isdir(base_dir)
        @warn "Results directory $(base_dir) does not exist. Run simulations first."
        return
    end
    
    println("Generating plots from existing results in $(base_dir)...")
    summarize_tasks(cfg, base_dir)
    generate_manuscript_plots(cfg, base_dir)
    println("Plot generation finished. Results saved to $(base_dir)")
end

# CLI entry point.
function main()
    mode, reset, replot, plot_only, workers, methods_to_run, rerun_methods = parse_args()
    
    # If --plot-only is set, only generate plots without running simulations
    if plot_only
        generate_plots_only(mode)
        return
    end
    
    cfg = default_simulation_config(mode; n_workers=workers)
    base_dir = joinpath(RESULTS_DIR, "simulation", string(cfg.mode))
    
    # If --reset is set, delete all demo and full directories before running
    if reset
        simulation_base = joinpath(RESULTS_DIR, "simulation")
        if isdir(simulation_base)
            demo_dir = joinpath(simulation_base, "demo")
            full_dir = joinpath(simulation_base, "full")
            
            if isdir(demo_dir)
                println("Deleting existing demo directory: $(demo_dir)")
                rm(demo_dir; recursive=true, force=true)
            end
            if isdir(full_dir)
                println("Deleting existing full directory: $(full_dir)")
                rm(full_dir; recursive=true, force=true)
            end
            println("Reset complete. All previous simulation results have been deleted.")
        end
    end
    
    ensure_results_dir(base_dir)
    
    # If --replot is set, delete existing plot directories to force regeneration
    if replot
        plots_dir = joinpath(base_dir, "plots")
        plots_manuscript_dir = joinpath(base_dir, "plots_manuscript")
        if isdir(plots_dir)
            rm(plots_dir; recursive=true, force=true)
            println("Deleted existing plots directory for regeneration.")
        end
        if isdir(plots_manuscript_dir)
            rm(plots_manuscript_dir; recursive=true, force=true)
            println("Deleted existing plots_manuscript directory for regeneration.")
        end
    end

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
    
    # Warn if user requested more workers than available threads
    if cfg.n_workers > nthreads()
        println("Warning: Requested $(cfg.n_workers) workers but Julia was started with only $(nthreads()) thread(s).")
        println("   To use more threads, restart Julia with: JULIA_NUM_THREADS=$(cfg.n_workers) julia simulation.jl ...")
        println("   Or on Windows PowerShell: \$env:JULIA_NUM_THREADS=$(cfg.n_workers); julia simulation.jl ...")
        println("   Proceeding with $(n_active_threads) thread(s)...")
        println()
    end
    
    # Display methods being run
    method_names = String[]
    if :nonlinear1 in methods_to_run
        push!(method_names, "NonLinear1")
    end
    if :nonlinear2 in methods_to_run
        push!(method_names, "NonLinear2")
    end
    if :coxph in methods_to_run
        push!(method_names, "CoxPH")
    end
    methods_str = isempty(method_names) ? "None" : join(method_names, ", ")
    
    println("=" ^ 80)
    println("Starting Simulation: $(string(cfg.mode))")
    println("  Scenario: g_type ∈ $(cfg.g_types), n ∈ $(cfg.n_values), replications = $(cfg.replications)")
    println("  Methods: $(methods_str)")
    println("  Workers: $(n_active_threads) thread(s)")
    println("=" ^ 80)
    println()
    
    limit = max(1, min(cfg.n_workers, nthreads()))
    sem = Base.Semaphore(limit)

    # Task-level progress bar via ProgressMeter.
    # Update on task completion; lock avoids interleaved progress output across threads.
    progress = Progress(length(tasks); desc="Simulations", dt=0.5)
    progress_lock = ReentrantLock()

    # Execute tasks grouped by (g_type, n), with parallel replications within each group
    println("\nStarting task execution...")
    
    for g_type in cfg.g_types
        println("\n" * "-" ^ 80)
        println("Processing scenario: g_type = $(g_type)")
        println("  Methods: $(methods_str)")
        println("-" ^ 80)
        for n in cfg.n_values
            # Collect all tasks for this (g_type, n) combination
            tasks_for_gn = [task for task in tasks if task.g_type == g_type && task.n == n]
            sort!(tasks_for_gn, by=t -> t.idx)
            
            total_for_scenario = length(tasks_for_gn)
            println("  Current scenario: g_type = $(g_type), n = $(n)")
            println("     Total tasks: $(total_for_scenario) replications")
            println("     Methods: $(methods_str)")
            
            # Execute replications in parallel for this (g_type, n) combination
            @sync begin
                for task in tasks_for_gn
                    Threads.@spawn begin
                        Base.acquire(sem)
                        try
                            run_single_task(task, cfg, base_dir, methods_to_run, rerun_methods)
                        catch e
                            # Print error details to help debug parallel execution issues
                            println(stderr, "\n[ERROR Task rep=$(task.idx)] Failed with error:")
                            println(stderr, "  Scenario: g_type=$(task.g_type), n=$(task.n)")
                            println(stderr, "  Error: ", e)
                            println(stderr, "  Stacktrace:")
                            showerror(stderr, e, catch_backtrace())
                            println(stderr)
                            rethrow(e)  # Re-throw to ensure task is marked as failed
                        finally
                            Base.release(sem)
                            lock(progress_lock) do
                                next!(progress)
                            end
                        end
                    end
                end
            end
            
            println("  Completed: $(total_for_scenario)/$(total_for_scenario) tasks for g_type=$(g_type), n=$(n)")
        end
        println("Completed all tasks for g_type = $(g_type)")
    end
    
    println("\nAll tasks completed: $(length(tasks))/$(length(tasks))")

    finish!(progress)

    println("Summarizing results...")
    summarize_tasks(cfg, base_dir)
    println("Generating plots...")
    generate_manuscript_plots(cfg, base_dir)
    println("Simulation finished. Results saved to $(base_dir)")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
