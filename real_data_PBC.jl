using Random                    # RNG for reproducibility across folds
using Distributions             # Prior/proposal distributions
using DataFrames                # Tabular data handling
using CSV                       # Reading/writing CSV files
using ProgressMeter             # Progress bar for cross-validation
using VectorizedStatistics      # Quantiles/means over arrays
using MLDataUtils               # K-fold splitting
using Serialization             # Placeholder for any future caching
using Statistics                # Basic stats utilities

include(joinpath(@__DIR__, "config.jl"))
include(joinpath(@__DIR__, "model.jl"))

using .Config
using .RJMCMCModel

# ---------------------------------------------------------------------
# Paths and settings
# ---------------------------------------------------------------------
cfg = real_data_config()
base_dir = joinpath(RESULTS_DIR, "real_data", "pbc")
if !isdir(base_dir)
    mkpath(base_dir)
end

# ---------------------------------------------------------------------
# Load data (CSV now placed in project root)
# ---------------------------------------------------------------------
df = CSV.read(joinpath(@__DIR__, "pbcData.csv"), DataFrame)

# ---------------------------------------------------------------------
# Preprocess covariates: standardize for better MCMC mixing
# ---------------------------------------------------------------------
Y, Delta, Z = df.Y, df.Delta, df.Z
X1 = (df.X1 .- mean(df.X1)) ./ std(df.X1)
X2 = (df.X2 .- mean(df.X2)) ./ std(df.X2)
X3 = (df.X3 .- mean(df.X3)) ./ std(df.X3)
X4 = (df.X4 .- mean(df.X4)) ./ std(df.X4)

X = hcat(X1, X2, X3, X4)
a, b = minimum(Z), maximum(Z)

# ---------------------------------------------------------------------
# Fit RJMCMC nonlinear model
# ---------------------------------------------------------------------
println("Running PBC analysis with ns=$(cfg.ns), burn-in=$(cfg.burn_in)")
results = RJMCMC_Nonlinear(X, Z, Delta, Y, a, b, cfg.mcmc_seed; ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX, Kmax=DEFAULT_KMAX)

# Posterior summaries for regression coefficients
betas_mean = mean(results["betas"], dims=2)
betas_lb = vquantile(results["betas"], 0.025, dims=2)
betas_ub = vquantile(results["betas"], 0.975, dims=2)

df_beta = DataFrame(Pos_Mean=betas_mean[:, 1], CrI_LB=betas_lb[:, 1], CrI_UB=betas_ub[:, 1])
CSV.write(joinpath(base_dir, "beta_PBC.csv"), df_beta)

# Posterior mean of g(z)
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

CSV.write(joinpath(base_dir, "g_PBC.csv"), DataFrame(z=z_grid, g=g_pos))

# ---------------------------------------------------------------------
# Cross-validated IBS for model comparison
# ---------------------------------------------------------------------
n_fold = cfg.folds
folds = kfolds(length(Y), n_fold)
IBS_nonlinear = zeros(n_fold)
IBS_coxph = zeros(n_fold)

@showprogress for i in 1:n_fold
    train_idx = folds[1][i]
    test_idx = folds[2][i]

    Y_train, Delta_train, X_train, Z_train = Y[train_idx], Delta[train_idx], X[train_idx, :], Z[train_idx]
    Y_test, Delta_test, X_test, Z_test = Y[test_idx], Delta[test_idx], X[test_idx, :], Z[test_idx]

    res_nonlinear = RJMCMC_Nonlinear(X_train, Z_train, Delta_train, Y_train, a, b, cfg.mcmc_seed; ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX, Kmax=DEFAULT_KMAX)
    res_coxph = RJMCMC_CoxPH(X_train, Z_train, Delta_train, Y_train, cfg.mcmc_seed; ns=cfg.ns, burn_in=cfg.burn_in, Hmax=DEFAULT_HMAX)

    IBS_nonlinear[i] = IBS(Y_train, Delta_train, X_train, Z_train, Y_test, Delta_test, X_test, Z_test, res_nonlinear, "nonlinear")
    IBS_coxph[i] = IBS(Y_train, Delta_train, X_train, Z_train, Y_test, Delta_test, X_test, Z_test, res_coxph, "coxph")
end

CSV.write(joinpath(base_dir, "IBS_PBC.csv"), DataFrame(Method=["Nonlinear", "CoxPH"], IBS=[mean(IBS_nonlinear), mean(IBS_coxph)]))
println("PBC analysis complete. Results stored in $(base_dir)")
