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
using StatsPlots
using CategoricalArrays

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "model.jl"))

using .Config
using .RJMCMCModel

struct SimulationTask
    g_type::String  # true g(z) shape
    n::Int          # sample size
    idx::Int        # replication id
end

# Parse command-line flags (--demo/--full/--reset/--replot/--plot-only/--workers=N).
function parse_args()
    mode = Config.full
    reset = false
    replot = false
    plot_only = false
    workers = nthreads()

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
        end
    end

    return mode, reset, replot, plot_only, workers
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

    # Use known support for Z to avoid boundary bias/kinks when plotting on full grid.
    a, b = cfg.z_min, cfg.z_max

    results_nonlinear = RJMCMC_Nonlinear(X_train, Z_train, Delta_train, Y_train, a, b, cfg.mcmc_seed; ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX, Kmax=DEFAULT_KMAX, Hcan=20, Kcan=20)
    results_coxph = RJMCMC_CoxPH(X_train, Z_train, Delta_train, Y_train, cfg.mcmc_seed; ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX, Hcan=20)

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

    # Evaluate Λ following notebook approach: only evaluate for t <= Tmax per iteration
    # Match notebook Cell 7: t_grid_reduced = t_grid[t_grid .<= Tmax]
    Lambda_nonlinear = zeros(length(t_grid))
    Lambda_nz_nonlinear = zeros(length(t_grid))
    for i in (bi + 1):ns
        H = Int(H_all[i])
        taus = taus_all[1:H, i]
        gammas = gammas_all[1:(H + 2), i]
        tau_i = Tmax  # Use Tmax as tau for this iteration
        
        # Only evaluate for points <= Tmax (matching notebook)
        valid_mask = t_grid .<= tau_i
        if any(valid_mask)
            t_grid_reduced = t_grid[valid_mask]
            est = Lambda_fun_est(t_grid_reduced, tau_i, taus, gammas)
            
            # Pad with zeros if needed (matching notebook logic)
            if length(est) < length(t_grid)
                est_full = zeros(length(t_grid))
                est_full[valid_mask] = est
            else
                est_full = est
            end
            
            Lambda_nonlinear .+= est_full
            Lambda_nz_nonlinear .+= valid_mask
        end
    end
    # Average only over iterations that contributed (matching notebook)
    Lambda_nonlinear = Lambda_nonlinear ./ max.(Lambda_nz_nonlinear, 1.0)
    Lambda_nonlinear = accumulate(max, Lambda_nonlinear)

    H_cox_all, taus_cox_all, gammas_cox_all = results_coxph["H"], results_coxph["taus"], results_coxph["gammas"]
    Lambda_coxph = zeros(length(t_grid))
    for i in (bi + 1):ns
        H = Int(H_cox_all[i])
        taus = taus_cox_all[1:H, i]
        gammas = gammas_cox_all[1:(H + 2), i]
        tau_i = Tmax  # Use Tmax as tau for this iteration
        
        # Evaluate for all points (CoxPH uses full t_grid in notebook)
        Lambda_coxph .+= Lambda_fun_est(t_grid, tau_i, taus, gammas)
    end
    Lambda_coxph ./= (ns - bi)
    Lambda_coxph = accumulate(max, Lambda_coxph)

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
        "mode" => string(cfg.mode),
    )

    open(results_path, "w") do file
        serialize(file, results_dict)
    end

    return :done
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

            g_samples = Vector{Vector{Float64}}()
            lambda_non_samples = Vector{Vector{Float64}}()
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
                push!(g_samples, results_dict["g_nonlinear"])
                push!(lambda_non_samples, results_dict["Lambda_nonlinear"])
                push!(lambda_cox_samples, results_dict["Lambda_coxph"])

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

                # --- Plotting (PDF) ---
                z_grid = cfg.z_grid
                t_grid = cfg.t_grid
                g_truth = g_true_values(g_type, z_grid)
                baseline = baseline_cumhaz(t_grid, cfg.hazard_a, cfg.hazard_b)

                if !isempty(g_samples)
                    g_mat = hcat(g_samples...)
                    g_mean = vec(mean(g_mat, dims=2))

                    plt_g = plot(z_grid, g_truth; lw=2, color=:blue, label="True")
                    plot!(plt_g, z_grid, g_mean; lw=2, color=:red, label="NonLinear")
                    xlabel!(plt_g, "z")
                    ylabel!(plt_g, "g(z)")
                    title!(plt_g, "n = $(n)")
                    savefig(plt_g, joinpath(plots_dir, "g_$(g_type)_n$(n).pdf"))
                end

                if !isempty(lambda_non_samples)
                    lam_non_mat = hcat(lambda_non_samples...)
                    lam_non_mean = vec(mean(lam_non_mat, dims=2))

                    plt_lam = plot(t_grid, baseline; lw=2, color=:blue, label="True")
                    plot!(plt_lam, t_grid, lam_non_mean; lw=2, color=:red, label="NonLinear")
                    xlabel!(plt_lam, "t")
                    ylabel!(plt_lam, "Λ(t)")
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
# -------------------------------------------------------------------------
function generate_manuscript_plots(cfg::SimulationConfig, base_dir::String)
    plot_dir = joinpath(base_dir, "plots_manuscript")
    mkpath(plot_dir)

    baseline = baseline_cumhaz(cfg.t_grid, cfg.hazard_a, cfg.hazard_b)
    ibs_rows = DataFrame(g_type=String[], n=Int[], Method=String[], IBS=Float64[])

    # Cache means per (g, n)
    lam_non_means = Dict{Tuple{String,Int}, Vector{Float64}}()
    lam_cox_means = Dict{Tuple{String,Int}, Union{Vector{Float64},Nothing}}()
    g_non_means = Dict{Tuple{String,Int}, Vector{Float64}}()

    for g_type in cfg.g_types
        for n in cfg.n_values
            lam_non_list = Vector{Vector{Float64}}()
            lam_cox_list = Vector{Vector{Float64}}()
            g_non_list = Vector{Vector{Float64}}()

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

                push!(lam_non_list, data["Lambda_nonlinear"])
                push!(g_non_list, data["g_nonlinear"])
                if haskey(data, "Lambda_coxph")
                    push!(lam_cox_list, data["Lambda_coxph"])
                end
                if haskey(data, "IBS_nonlinear")
                    push!(ibs_rows, (g_type, n, "NonLinear", data["IBS_nonlinear"]))
                end
                if haskey(data, "IBS_coxph")
                    push!(ibs_rows, (g_type, n, "CoxPH", data["IBS_coxph"]))
                end
            end

            if !found_any
                @warn "No results found for g=$(g_type), n=$(n) in $(base_dir); plots may be empty. Rerun simulations for this setting."
                continue
            end

            if !isempty(lam_non_list)
                lam_non_means[(g_type, n)] = vec(mean(hcat(lam_non_list...), dims=2))
            end
            if !isempty(lam_cox_list)
                lam_cox_means[(g_type, n)] = vec(mean(hcat(lam_cox_list...), dims=2))
            else
                lam_cox_means[(g_type, n)] = nothing
            end
            if !isempty(g_non_list)
                g_non_means[(g_type, n)] = vec(mean(hcat(g_non_list...), dims=2))
            end
        end
    end

    # Figures for each g_type
    for g_type in cfg.g_types
        # Collect available n values for this g_type
        available_n_lam = [n for n in cfg.n_values if haskey(lam_non_means, (g_type, n))]
        available_n_g = [n for n in cfg.n_values if haskey(g_non_means, (g_type, n))]
        
        # Λ panels - only plot available n values
        if !isempty(available_n_lam)
            n_lam_count = length(available_n_lam)
            lam_layout = (1, n_lam_count)
            p_lam = plot(layout=lam_layout, size=(400 * n_lam_count, 400), legend=:topright)
            for (i, n) in enumerate(available_n_lam)
                lam_non = lam_non_means[(g_type, n)]
                plt = plot(cfg.t_grid, baseline; color=:black, lw=2, label=i == 1 ? "True" : "", legend=:topright)
                plot!(plt, cfg.t_grid, lam_non; color=:red, lw=2, label=i == 1 ? "NonLinear" : "", linestyle=:dash)
                title!(plt, "n = $(n)")
                xlabel!(plt, "t")
                ylabel!(plt, "Λ(t)")
                plot!(p_lam, plt, subplot=i)
            end
            savefig(p_lam, joinpath(plot_dir, "Lambda_$(g_type).pdf"))
            println("Generated Lambda plot for g=$(g_type) with $(n_lam_count) sample size(s): $(available_n_lam)")
        else
            @warn "No Lambda data available for g=$(g_type); skipping plot."
        end

        # g(z) panels (only NonLinear estimated) - only plot available n values
        if !isempty(available_n_g)
            n_g_count = length(available_n_g)
            g_layout = (1, n_g_count)
            p_g = plot(layout=g_layout, size=(400 * n_g_count, 400), legend=:topright)
            g_true = g_true_values(g_type, cfg.z_grid)
            for (i, n) in enumerate(available_n_g)
                g_non = g_non_means[(g_type, n)]
                plt = plot(cfg.z_grid, g_true; color=:black, lw=2, label=i == 1 ? "True" : "", legend=:topright)
                plot!(plt, cfg.z_grid, g_non; color=:red, lw=2, label=i == 1 ? "NonLinear" : "", linestyle=:dash)
                title!(plt, "n = $(n)")
                xlabel!(plt, "z")
                ylabel!(plt, "g(z)")
                plot!(p_g, plt, subplot=i)
            end
            savefig(p_g, joinpath(plot_dir, "g_$(g_type).pdf"))
            println("Generated g(z) plot for g=$(g_type) with $(n_g_count) sample size(s): $(available_n_g)")
        else
            @warn "No g(z) data available for g=$(g_type); skipping plot."
        end
    end

    # IBS boxplot (Figure 4 style) - only plot g_types with data
    if !isempty(ibs_rows)
        df_ibs = ibs_rows
        df_ibs.n_str = string.(df_ibs.n)
        df_ibs.Method = categorical(df_ibs.Method, ordered=true, levels=["CoxPH", "NonLinear"])
        
        # Collect g_types that have IBS data
        available_g_types_ibs = unique(df_ibs.g_type)
        if !isempty(available_g_types_ibs)
            n_g_types = length(available_g_types_ibs)
            p_ibs = plot(layout=(1, n_g_types), size=(600 * n_g_types, 500), legend=:topright)
            colors = [:pink, :teal]
            for (i, g_type) in enumerate(available_g_types_ibs)
                subdf = df_ibs[df_ibs.g_type .== g_type, :]
                if nrow(subdf) > 0
                    plt = @df subdf boxplot(:n_str, :IBS, group=:Method, legend=i == 1 ? :topright : false, palette=colors)
                    title!(plt, string(uppercasefirst(g_type)))
                    xlabel!(plt, "n")
                    ylabel!(plt, "IBS")
                    plot!(p_ibs, plt, subplot=i)
                end
            end
            savefig(p_ibs, joinpath(plot_dir, "IBS_boxplots.pdf"))
            println("Generated IBS boxplot with $(n_g_types) g_type(s): $(available_g_types_ibs)")
        else
            @warn "No IBS data available for plotting."
        end
    else
        @warn "IBS table empty; rerun simulations to populate IBS metrics."
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
    mode, reset, replot, plot_only, workers = parse_args()
    
    # If --plot-only is set, only generate plots without running simulations
    if plot_only
        generate_plots_only(mode)
        return
    end
    
    cfg = default_simulation_config(mode; n_workers=workers)
    base_dir = joinpath(RESULTS_DIR, "simulation", string(cfg.mode))

    if reset && isdir(base_dir)
        rm(base_dir; recursive=true, force=true)
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
    println("Running simulations with $n_active_threads threads (set JULIA_NUM_THREADS to change).")
    prog = Progress(length(tasks); desc="Simulations")
    progress_lock = ReentrantLock()
    limit = max(1, min(cfg.n_workers, nthreads()))
    sem = Base.Semaphore(limit)

    @sync begin
        for task in tasks
            Threads.@spawn begin
                Base.acquire(sem)
                try
                    run_single_task(task, cfg, base_dir)
                    lock(progress_lock) do
                        next!(prog)
                    end
                finally
                    Base.release(sem)
                end
            end
        end
    end

    summarize_tasks(cfg, base_dir)
    generate_manuscript_plots(cfg, base_dir)
    println("Simulation finished. Results saved to $(base_dir)")
end

main()
