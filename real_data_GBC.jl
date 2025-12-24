using Random                    # RNG for reproducibility across folds
using Distributions             # Prior/proposal distributions
using DataFrames                # Tabular data handling
using CSV                       # Reading/writing CSV files
using ProgressMeter             # Progress bar for cross-validation
using VectorizedStatistics      # Quantiles/means over arrays
using MLDataUtils               # K-fold splitting
using Serialization             # Placeholder for any future caching
using Statistics                # Basic stats utilities
using Plots                     # Plotting
using LaTeXStrings              # LaTeX labels

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "model.jl"))

using .Config
using .RJMCMCModel

# ---------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------
cfg = real_data_config()
base_dir = joinpath(RESULTS_DIR, "real_data", "gbc")
if !isdir(base_dir)
    mkpath(base_dir)
end

# ---------------------------------------------------------------------
# Load data (CSV now placed in project root)
# ---------------------------------------------------------------------
df = CSV.read(joinpath(@__DIR__, "gbcData.csv"), DataFrame)

# ---------------------------------------------------------------------
# Preprocess covariates: standardize for better MCMC mixing
# ---------------------------------------------------------------------
Y, Delta = df.Y, df.Delta
X1 = (df.X1 .- mean(df.X1)) ./ std(df.X1)
X2 = (df.X2 .- mean(df.X2)) ./ std(df.X2)
X3 = (df.X3 .- mean(df.X3)) ./ std(df.X3)
X4 = (df.X4 .- mean(df.X4)) ./ std(df.X4)
Z = (df.Z .- mean(df.Z)) ./ std(df.Z)

X = hcat(X1, X2, X3, X4)
a, b = minimum(Z), maximum(Z)

# ---------------------------------------------------------------------
# Fit RJMCMC nonlinear model
# ---------------------------------------------------------------------
println("Running GBC analysis with ns=$(cfg.ns), burn-in=$(cfg.burn_in)")
results = RJMCMC_Nonlinear(X, Z, Delta, Y, a, b, cfg.mcmc_seed; ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX, Kmax=DEFAULT_KMAX)
results_dirichlet = RJMCMC_Nonlinear_Dirichlet(X, Z, Delta, Y, a, b, cfg.mcmc_seed; ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX, Kmax=DEFAULT_KMAX)
results_coxph = RJMCMC_CoxPH(X, Z, Delta, Y, cfg.mcmc_seed; ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX)

# Store tau/zeta/H/K traces for NonLinear1 and NonLinear2
function long_matrix_df(mat, method::String, value_name::Symbol)
    n_rows, n_cols = size(mat)
    df = DataFrame(Method=fill(method, n_rows * n_cols),
        Iter=repeat(1:n_cols, inner=n_rows),
        Index=repeat(1:n_rows, n_cols),
        Value=vec(mat))
    rename!(df, :Value => value_name)
    return df
end

function hk_df(H_all, K_all, method::String)
    n = length(H_all)
    return DataFrame(Method=fill(method, n),
        Iter=1:n,
        H=vec(H_all),
        K=vec(K_all))
end

df_tau = vcat(
    long_matrix_df(results["taus"], "NonLinear1", :Tau),
    long_matrix_df(results_dirichlet["taus"], "NonLinear2", :Tau)
)
df_zeta = vcat(
    long_matrix_df(results["zetas"], "NonLinear1", :Zeta),
    long_matrix_df(results_dirichlet["zetas"], "NonLinear2", :Zeta)
)
df_hk = vcat(
    hk_df(results["H"], results["K"], "NonLinear1"),
    hk_df(results_dirichlet["H"], results_dirichlet["K"], "NonLinear2")
)

CSV.write(joinpath(base_dir, "tau_GBC.csv"), df_tau)
CSV.write(joinpath(base_dir, "zeta_GBC.csv"), df_zeta)
CSV.write(joinpath(base_dir, "HK_GBC.csv"), df_hk)

# Posterior summaries for regression coefficients
function summarize_betas(betas_all, method::String)
    betas_mean = vec(mean(betas_all, dims=2))
    betas_lb = vec(vquantile(betas_all, 0.025, dims=2))
    betas_ub = vec(vquantile(betas_all, 0.975, dims=2))
    return DataFrame(Method=fill(method, length(betas_mean)),
        Beta=1:length(betas_mean),
        Pos_Mean=betas_mean, CrI_LB=betas_lb, CrI_UB=betas_ub)
end

df_beta = vcat(
    summarize_betas(results["betas"], "NonLinear1"),
    summarize_betas(results_dirichlet["betas"], "NonLinear2"),
    summarize_betas(results_coxph["betas"], "CoxPH")
)
CSV.write(joinpath(base_dir, "beta_GBC.csv"), df_beta)

# Posterior mean of g(z) for NonLinear1
K_all, zetas_all, xis_all = results["K"], results["zetas"], results["xis"]
ns, bi = results["ns"], results["burn_in"]
z_grid = range(a; stop=b, length=100)

g_pos = zeros(length(z_grid))
for i in (bi + 1):ns
    K = Int(K_all[i])
    zetas = zetas_all[1:K, i]
    xis = xis_all[1:(K + 2), i]
    g_pos .+= g_fun_est(z_grid, a, b, zetas, xis)
end
g_pos ./= (ns - bi)

# Posterior mean of g(z) for NonLinear2 (Dirichlet-Gamma prior)
K_all_dir, zetas_all_dir, xis_all_dir = results_dirichlet["K"], results_dirichlet["zetas"], results_dirichlet["xis"]
ns_dir, bi_dir = results_dirichlet["ns"], results_dirichlet["burn_in"]

g_pos_dir = zeros(length(z_grid))
for i in (bi_dir + 1):ns_dir
    K = Int(K_all_dir[i])
    zetas = zetas_all_dir[1:K, i]
    xis = xis_all_dir[1:(K + 2), i]
    g_pos_dir .+= g_fun_est(z_grid, a, b, zetas, xis)
end
g_pos_dir ./= (ns_dir - bi_dir)

CSV.write(joinpath(base_dir, "g_GBC.csv"), DataFrame(z=z_grid, g_non1=g_pos, g_non2=g_pos_dir))

# Plot NonLinear1 vs NonLinear2 (colors match simulation.jl)
plt_g = plot(z_grid, g_pos; lw=2, color=:red, linestyle=:dot, label="NonLinear1",
    xlabel="Standardized age", ylabel=L"$g(z)$")
plot!(plt_g, z_grid, g_pos_dir; lw=2, color=:green, linestyle=:dash, label="NonLinear2")
savefig(plt_g, joinpath(base_dir, "g_compare_GBC.pdf"))

# Histogram of Z values used in nonlinear term
plt_z = histogram(Z; bins=30, color=:gray, xlabel="Standardized age", ylabel="Count")
savefig(plt_z, joinpath(base_dir, "z_hist_GBC.pdf"))

# ---------------------------------------------------------------------
# Cross-validated IBS for model comparison
# ---------------------------------------------------------------------
n_fold = cfg.folds
folds = kfolds(length(Y), n_fold)
IBS_nonlinear1 = zeros(n_fold)
IBS_nonlinear2 = zeros(n_fold)
IBS_coxph = zeros(n_fold)

@showprogress for i in 1:n_fold
    train_idx = folds[1][i]
    test_idx = folds[2][i]

    Y_train, Delta_train, X_train, Z_train = Y[train_idx], Delta[train_idx], X[train_idx, :], Z[train_idx]
    Y_test, Delta_test, X_test, Z_test = Y[test_idx], Delta[test_idx], X[test_idx, :], Z[test_idx]

    res_nonlinear1 = RJMCMC_Nonlinear(X_train, Z_train, Delta_train, Y_train, a, b, cfg.mcmc_seed; ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX, Kmax=DEFAULT_KMAX)
    res_nonlinear2 = RJMCMC_Nonlinear_Dirichlet(X_train, Z_train, Delta_train, Y_train, a, b, cfg.mcmc_seed; ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX, Kmax=DEFAULT_KMAX)
    res_coxph = RJMCMC_CoxPH(X_train, Z_train, Delta_train, Y_train, cfg.mcmc_seed; ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX)

    IBS_nonlinear1[i] = IBS(Y_train, Delta_train, X_train, Z_train, Y_test, Delta_test, X_test, Z_test, res_nonlinear1, "nonlinear")
    IBS_nonlinear2[i] = IBS(Y_train, Delta_train, X_train, Z_train, Y_test, Delta_test, X_test, Z_test, res_nonlinear2, "nonlinear")
    IBS_coxph[i] = IBS(Y_train, Delta_train, X_train, Z_train, Y_test, Delta_test, X_test, Z_test, res_coxph, "coxph")
end

CSV.write(joinpath(base_dir, "IBS_GBC.csv"),
    DataFrame(Method=["NonLinear1", "NonLinear2", "CoxPH"],
              IBS=[mean(IBS_nonlinear1), mean(IBS_nonlinear2), mean(IBS_coxph)]))
println("GBC analysis complete. Results stored in $(base_dir)")
