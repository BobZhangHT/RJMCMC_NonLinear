module RJMCMCModel

# Core RJMCMC samplers and survival utilities shared by simulation and real-data scripts.

using Random
using Distributions
using Statistics
using LinearAlgebra
using ProgressMeter
using VectorizedStatistics
using StatsBase
using SpecialFunctions  # For loggamma in Dirichlet prior

export RJMCMC_Nonlinear,
       RJMCMC_Nonlinear_Dirichlet,
       RJMCMC_CoxPH,
       St_pred,
       coxph_St_pred,
       IBS,
       KM_est,
       loglambda_fun_est,
       g_fun_est,
       Lambda_fun_est,
       DEFAULT_NS,
       DEFAULT_BURN_IN,
       DEFAULT_HMAX,
       DEFAULT_KMAX

const DEFAULT_NS = 5000
const DEFAULT_BURN_IN = DEFAULT_NS ÷ 2
const DEFAULT_HMAX = 10
const DEFAULT_KMAX = 10

@inline log2π() = 1.8378770664093453 # log(2π)

@inline function logpdf_normal(x::Float64, μ::Float64, σ::Float64)
    invσ = 1.0 / σ
    z = (x - μ) * invσ
    return -0.5 * (z * z + log2π()) - log(σ)
end

mutable struct LikelihoodWorkspace{T<:Float64}
    eta::Vector{T}          # X * betas (then reused as η = Xβ + g)
    knots_h::Vector{T}      # baseline knots (0, taus..., tau)
    cumhaz_h::Vector{T}     # cumulative ∫ exp(logλ) up to each knot
    knots_g::Vector{T}      # g(z) knots (a, zetas..., b)
    g_values::Vector{T}     # g(z) values (pre-allocated)
    loglambda_values::Vector{T}  # logλ(Y) values (pre-allocated)
    Lambda_values::Vector{T}     # Λ(Y) values (pre-allocated)
    Xbeta_plus_g::Vector{T}      # Xβ + g (pre-allocated)
end

function LikelihoodWorkspace(n::Integer, Hmax::Integer, Kmax::Integer)
    LikelihoodWorkspace(
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, Hmax + 2),
        Vector{Float64}(undef, Hmax + 2),
        Vector{Float64}(undef, Kmax + 2),
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n),
    )
end

@inline function fill_knots!(buf::Vector{Float64}, left::Float64, mids::AbstractVector{<:Real}, right::Float64)
    H = length(mids)
    buf[1] = left
    @inbounds for i in 1:H
        buf[i + 1] = Float64(mids[i])
    end
    buf[H + 2] = right
    return H + 2
end

@inline function searchsortedfirst_len(v::Vector{Float64}, len::Int, x::Float64)
    @inbounds if x > v[len]
        return len + 1
    end
    lo = 1
    hi = len
    while lo < hi
        mid = (lo + hi) >>> 1
        @inbounds if v[mid] < x
            lo = mid + 1
        else
            hi = mid
        end
    end
    return lo
end

@inline function piecewise_linear_at(x::Float64, knots::Vector{Float64}, values::AbstractVector{<:Real}, len::Int)
    i = searchsortedfirst_len(knots, len, x)
    if i <= 1
        return 0.0
    elseif i > len
        return 0.0
    end
    j = i - 1
    x0 = knots[j]
    x1 = knots[j + 1]
    y0 = Float64(values[j])
    y1 = Float64(values[j + 1])
    return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
end

@inline function prepare_cumhaz!(cumhaz::Vector{Float64}, knots::Vector{Float64}, gammas::AbstractVector{<:Real}, len::Int)
    cumhaz[1] = 0.0
    @inbounds for j in 1:(len - 1)
        x0 = knots[j]
        x1 = knots[j + 1]
        y0 = Float64(gammas[j])
        y1 = Float64(gammas[j + 1])
        dx = x1 - x0
        dy = y1 - y0
        seg = if abs(dy) < 1e-12
            exp(y0) * dx
        else
            dx * (exp(y1) - exp(y0)) / dy
        end
        cumhaz[j + 1] = cumhaz[j] + seg
    end
    return nothing
end

@inline function cumhaz_at(x::Float64, knots::Vector{Float64}, cumhaz::Vector{Float64}, gammas::AbstractVector{<:Real}, len::Int)
    i = searchsortedfirst_len(knots, len, x)
    if i <= 1
        return 0.0
    elseif i > len
        return 0.0
    end
    j = i - 1
    x0 = knots[j]
    x1 = knots[j + 1]
    y0 = Float64(gammas[j])
    y1 = Float64(gammas[j + 1])
    dx = x1 - x0
    dy = y1 - y0
    s = dy / dx
    u = x - x0
    part = if abs(s) < 1e-12
        exp(y0) * u
    else
        exp(y0) * (expm1(s * u)) / s
    end
    return cumhaz[j] + part
end

# Piecewise log-baseline hazard evaluated at times t given knot locations taus and slopes gammas.
function loglambda_fun_est(t, tau, taus, gammas)
    out = Vector{Float64}(undef, length(t))
    H = length(taus)
    if H == 0
        slope = (Float64(gammas[2]) - Float64(gammas[1])) / Float64(tau)
        γ1 = Float64(gammas[1])
        @inbounds for i in eachindex(t)
            out[i] = γ1 + slope * Float64(t[i])
        end
        return out
    end
    knots = Vector{Float64}(undef, H + 2)
    fill_knots!(knots, 0.0, taus, Float64(tau))
    @inbounds for i in eachindex(t)
        out[i] = piecewise_linear_at(Float64(t[i]), knots, gammas, H + 2)
    end
    return out
end

function loglambda_fun_est!(out::Vector{Float64}, t, tau, taus, gammas, knots_buf::Vector{Float64})
    H = length(taus)
    if H == 0
        slope = (Float64(gammas[2]) - Float64(gammas[1])) / Float64(tau)
        γ1 = Float64(gammas[1])
        @inbounds for i in eachindex(out, t)
            out[i] = γ1 + slope * Float64(t[i])
        end
        return out
    end
    len = fill_knots!(knots_buf, 0.0, taus, Float64(tau))
    @inbounds for i in eachindex(out, t)
        out[i] = piecewise_linear_at(Float64(t[i]), knots_buf, gammas, len)
    end
    return out
end

# Piecewise-linear nonlinear covariate effect g(z) over [a, b].
function g_fun_est(z, a, b, zetas, xis)
    out = Vector{Float64}(undef, length(z))
    K = length(zetas)
    if K == 0
        slope = (Float64(xis[2]) - Float64(xis[1])) / (Float64(b) - Float64(a))
        ξ1 = Float64(xis[1])
        a_ = Float64(a)
        @inbounds for i in eachindex(z)
            out[i] = ξ1 + slope * (Float64(z[i]) - a_)
        end
        return out
    end
    knots = Vector{Float64}(undef, K + 2)
    fill_knots!(knots, Float64(a), zetas, Float64(b))
    @inbounds for i in eachindex(z)
        out[i] = piecewise_linear_at(Float64(z[i]), knots, xis, K + 2)
    end
    return out
end

function g_fun_est!(out::Vector{Float64}, z, a, b, zetas, xis, knots_buf::Vector{Float64})
    K = length(zetas)
    if K == 0
        slope = (Float64(xis[2]) - Float64(xis[1])) / (Float64(b) - Float64(a))
        ξ1 = Float64(xis[1])
        a_ = Float64(a)
        @inbounds for i in eachindex(out, z)
            out[i] = ξ1 + slope * (Float64(z[i]) - a_)
        end
        return out
    end
    len = fill_knots!(knots_buf, Float64(a), zetas, Float64(b))
    @inbounds for i in eachindex(out, z)
        out[i] = piecewise_linear_at(Float64(z[i]), knots_buf, xis, len)
    end
    return out
end

# Integrated hazard Λ(t) corresponding to the piecewise log-λ definition.
function Lambda_fun_est(t, tau, taus, gammas)
    # Match historical formulation (model1229): clamp t to [0, tau] to avoid extrapolation.
    out = Vector{Float64}(undef, length(t))
    H = length(taus)
    knots = Vector{Float64}(undef, H + 2)
    fill_knots!(knots, 0.0, taus, Float64(tau))
    cumhaz = Vector{Float64}(undef, H + 2)
    prepare_cumhaz!(cumhaz, knots, gammas, H + 2)
    @inbounds for i in eachindex(t)
        ti = Float64(t[i])
        if ti > tau
            ti = Float64(tau)
        end
        out[i] = cumhaz_at(ti, knots, cumhaz, gammas, H + 2)
    end
    return out
end

function Lambda_fun_est!(out::Vector{Float64}, t, tau, taus, gammas, knots_buf::Vector{Float64}, cumhaz_buf::Vector{Float64})
    H = length(taus)
    len = fill_knots!(knots_buf, 0.0, taus, Float64(tau))
    prepare_cumhaz!(cumhaz_buf, knots_buf, gammas, len)
    @inbounds for i in eachindex(out, t)
        ti = Float64(t[i])
        if ti > tau
            ti = Float64(tau)
        end
        out[i] = cumhaz_at(ti, knots_buf, cumhaz_buf, gammas, len)
    end
    return out
end

# Full log-likelihood for nonlinear model (baseline + covariates + nonlinear g(Z)).
function loglkh_cal(X, Z, Delta, Y,
                    betas,
                    tau, taus, gammas,
                    a, b, zetas, xis)
    ws = LikelihoodWorkspace(length(Y), length(taus) + 2, length(zetas) + 2)
    return loglkh_cal(ws, X, Z, Delta, Y, betas, tau, taus, gammas, a, b, zetas, xis)
end

function loglkh_cal(ws::LikelihoodWorkspace,
                    X, Z, Delta, Y,
                    betas,
                    tau, taus, gammas,
                    a, b, zetas, xis)
    # Optimized vectorized version - reuse workspace and use in-place operations
    mul!(ws.eta, X, betas)
    
    # Compute function values using in-place optimized functions
    g_fun_est!(ws.g_values, Z, a, b, zetas, xis, ws.knots_g)
    loglambda_fun_est!(ws.loglambda_values, Y, tau, taus, gammas, ws.knots_h)
    Lambda_fun_est!(ws.Lambda_values, Y, tau, taus, gammas, ws.knots_h, ws.cumhaz_h)
    
    # Optimized vectorized log-likelihood calculation - avoid intermediate allocations
    @inbounds @simd for i in eachindex(ws.eta)
        ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
    end
    # Use optimized loop instead of broadcasting to avoid temporary arrays
    loglkh = 0.0
    @inbounds @simd for i in eachindex(Delta)
        exp_val = exp(ws.Xbeta_plus_g[i])
        loglkh += Delta[i] * (ws.loglambda_values[i] + ws.Xbeta_plus_g[i]) - 
                  ws.Lambda_values[i] * exp_val
    end
    return loglkh
end

# Incremental log-likelihood update for Beta change (only beta_j changes)
function loglkh_cal_beta_inc(ws::LikelihoodWorkspace,
                             X, Z, Delta, Y,
                             betas_old, beta_j_new, j,
                             tau, taus, gammas,
                             a, b, zetas, xis)
    # Only update Xbeta for changed beta_j
    beta_diff = beta_j_new - betas_old[j]
    @inbounds @simd for i in eachindex(ws.eta)
        ws.eta[i] += X[i, j] * beta_diff
        ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
    end
    # Recompute log-likelihood (only this part changes) - optimized vectorized version
    loglkh = 0.0
    @inbounds @simd for i in eachindex(Delta)
        exp_val = exp(ws.Xbeta_plus_g[i])
        loglkh += Delta[i] * (ws.loglambda_values[i] + ws.Xbeta_plus_g[i]) - 
                  ws.Lambda_values[i] * exp_val
    end
    return loglkh
end

# Incremental log-likelihood update for Xi change (only xi_k changes)
function loglkh_cal_xi_inc(ws::LikelihoodWorkspace,
                          X, Z, Delta, Y,
                          betas,
                          tau, taus, gammas,
                          a, b, zetas, xis_new,
                          ws_g_old::Vector{Float64})
    # Only recompute g_values (xi changed)
    g_fun_est!(ws.g_values, Z, a, b, zetas, xis_new, ws.knots_g)
    # Update Xbeta_plus_g
    @inbounds @simd for i in eachindex(ws.eta)
        ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
    end
    # Recompute log-likelihood - optimized vectorized version
    loglkh = 0.0
    @inbounds @simd for i in eachindex(Delta)
        exp_val = exp(ws.Xbeta_plus_g[i])
        loglkh += Delta[i] * (ws.loglambda_values[i] + ws.Xbeta_plus_g[i]) - 
                  ws.Lambda_values[i] * exp_val
    end
    return loglkh
end

# Optimized helper function to compute log-likelihood from pre-computed values
@inline function compute_loglkh_optimized(Delta::AbstractVector, loglambda_values::AbstractVector, 
                                          Xbeta_plus_g::AbstractVector, Lambda_values::AbstractVector)
    loglkh = 0.0
    @inbounds @simd for i in eachindex(Delta)
        exp_val = exp(Xbeta_plus_g[i])
        loglkh += Delta[i] * (loglambda_values[i] + Xbeta_plus_g[i]) - Lambda_values[i] * exp_val
    end
    return loglkh
end

# Optimized helper function for CoxPH model
@inline function compute_coxph_loglkh_optimized(Delta::AbstractVector, loglambda_values::AbstractVector, 
                                               eta::AbstractVector, Lambda_values::AbstractVector)
    loglkh = 0.0
    @inbounds @simd for i in eachindex(Delta)
        exp_val = exp(eta[i])
        loglkh += Delta[i] * (loglambda_values[i] + eta[i]) - Lambda_values[i] * exp_val
    end
    return loglkh
end

# Reversible-jump sampler for the nonlinear PH model (baseline + piecewise g(z)).
function RJMCMC_Nonlinear(
        X, Z, Delta, Y, a, b,
        random_seed;
        ns::Int=DEFAULT_NS,
        burn_in::Int=DEFAULT_BURN_IN,
        RJMCMC_indicator::Bool=true,
        Adapt_C::Bool=true,
        Hmax::Int=DEFAULT_HMAX,
        Kmax::Int=DEFAULT_KMAX,
        show_progress::Bool=true
     )

    NS = ns
    BI = burn_in
    # ---------------------------- Hyper-Parameters ----------------------------
    # dimension of beta
    dim_beta = size(X,2)
    ws = LikelihoodWorkspace(length(Y), Hmax, Kmax)
    
    # hyper-parameters
    # obtain the observed time
    obs_time = (Y .* Delta) 
    obs_time = obs_time[obs_time .> 0]
    tau  = maximum(obs_time)
    
    # prior parameter for number of probability
    mu_H = mu_K = 1

    # initial H
    H = 0
    # initial K
    K = 0
    
    # coefficients
    c_gamma = 1    # for gamma
    c_xi = 1
    c_beta = 1     # for betas
    c_cnt = 0
    AHigh = 0.4
    ALow = 0.2
    
    # number of reports
    n_report = 250
    
    # birth probability
    r_HB = 0.5
    r_KB = 0.5
    
    # ---------------------------- Coefficient Variables ----------------------------
    # PH Model coefficients
    betas_all = zeros(dim_beta,NS)
    
    # hazard function coefficients
    H_all = zeros(NS)
    taus_all = zeros(Hmax,NS)     # points
    gammas_all = zeros(Hmax+2,NS) # slope
    
    # smoothing function coefficients
    K_all = zeros(NS)
    zetas_all = zeros(Kmax,NS)  # points
    xis_all = zeros(Kmax+2,NS)  # slope
    
    # variance 
    sigmas_gamma_all = zeros(NS)
    sigmas_xi_all = zeros(NS)
    
    # ---------------------------- Acceptance Variables ----------------------------
    # acceptance rate calculation
    
    # PH Model coefficients
    abetas_all = zeros(dim_beta,NS)
    
    # hazard function coefficients
    aH_all = zeros(NS)
    agammas_all = zeros(Hmax+2,NS)  # slope
    
    # smoothing function coefficients
    aK_all = zeros(NS)
    axis_all = zeros(Kmax+2,NS)     # slope
    
    # ------------------ Prior Specification ------------------ 
    
    Random.seed!(random_seed)
    
    # Create candidate sets (matching model1229.jl exactly)
    Hcan = 20  # Candidate set size for baseline hazard knots
    Kcan = 20  # Candidate set size for g(z) knots
    taus_can = LinRange(0, tau, Hcan+2)[2:(Hcan+1)]
    zetas_can = LinRange(a, b, Kcan+2)[2:(Kcan+1)]
	
    # prior: baseline hazard
    # Use candidate set method (matching model1229.jl exactly)
    if H > 0
        taus = sort(sample(taus_can, H, replace=false))
    else
        taus = Float64[]  # Empty when H=0
    end
    taus_ = [0 taus' tau][1,:]
    
    gammas = zeros(H+2)
    gammas[1] = rand(Normal(0,5))
    for i in 2:(H+2)
        gammas[i] = rand(Normal(gammas[i-1],1))
    end  
    
    # prior: smooth function
    # Use candidate set method (matching model1229.jl exactly)
    if K > 0
        zetas = sort(sample(zetas_can, K, replace=false))
    else
        zetas = Float64[]  # Empty when K=0
    end
    zetas_ = [a zetas' b][1,:]
    
    xis = zeros(K+2)
    xis[1] = 0
    for i in 2:(K+2)
        xis[i] = rand(Normal(xis[i-1],1))
    end
    
    # beta: coefficients
    betas = zeros(dim_beta)
    
    # ------------------ Update Coefficient ------------------ 
    H_all[1] = H
    taus_all[1:H,1] = taus
    gammas_all[1:(H+2),1] = gammas
    
    K_all[1] = K
    zetas_all[1:K,1] = zetas
    xis_all[1:(K+2),1] = xis
    
    betas_all[:,1] = betas
    
    sigmas_gamma_all[1] = 1
    sigmas_xi_all[1] = 1
    
    # ------------------ Update Acceptance Count ------------------ 
    aH_all[1] = 1

    agammas_all[1:(H+2),1] .= 1
    
    aK_all[1] = 1 
    axis_all[1:(K+2),1] .= 1  
    abetas_all[:,1] .= 1

    Random.seed!(random_seed)

    # ---------------------------- MCMC Loop ----------------------------
    # ---------------------------- MCMC Loop ----------------------------
    # Only create progress bar if explicitly enabled
    if show_progress
        prog = Progress(NS - 1; desc="MCMC")
    end
    for iter in 2:NS

        # ----------------------------- Update Gamma ----------------------------- 
        gammas_star = copy(gammas)
        agamma_vec = zeros(size(gammas)[1])
        sigma_gamma = sigmas_gamma_all[iter-1]
        
        # Pre-compute components that don't change during gamma updates
        mul!(ws.eta, X, betas)
        g_fun_est!(ws.g_values, Z, a, b, zetas, xis, ws.knots_g)
        
        for h in 1:(H+2)        
            # compute the denominator - use optimized logpdf_normal instead of log(pdf(...))
            log_prob_de = logpdf_normal(gammas_star[1], 0.0, 5.0)
            @inbounds for hh in 2:(H+2)
                log_prob_de += logpdf_normal(gammas_star[hh], gammas_star[hh-1], sigma_gamma)
            end
            
            # Compute loglambda and Lambda for current gammas
            loglambda_fun_est!(ws.loglambda_values, Y, tau, taus, gammas_star, ws.knots_h)
            Lambda_fun_est!(ws.Lambda_values, Y, tau, taus, gammas_star, ws.knots_h, ws.cumhaz_h)
            @inbounds @simd for i in eachindex(ws.eta)
                ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
            end
            log_de = log_prob_de + compute_loglkh_optimized(Delta, ws.loglambda_values, ws.Xbeta_plus_g, ws.Lambda_values)
            
            # compute the numerator
            # propose a new gamma
            gamma_h_new = rand(Uniform(gammas[h]-c_gamma, gammas[h]+c_gamma))
            gammas_star[h] = gamma_h_new
            
            log_prob_num = logpdf_normal(gammas_star[1], 0.0, 5.0)
            @inbounds for hh in 2:(H+2)
                log_prob_num += logpdf_normal(gammas_star[hh], gammas_star[hh-1], sigma_gamma)
            end
            
            # Recompute loglambda and Lambda for proposed gammas
            loglambda_fun_est!(ws.loglambda_values, Y, tau, taus, gammas_star, ws.knots_h)
            Lambda_fun_est!(ws.Lambda_values, Y, tau, taus, gammas_star, ws.knots_h, ws.cumhaz_h)
            @inbounds @simd for i in eachindex(ws.eta)
                ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
            end
            log_num = log_prob_num + compute_loglkh_optimized(Delta, ws.loglambda_values, ws.Xbeta_plus_g, ws.Lambda_values)
            
            # acceptance ratio
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            # accept or not
            agamma_vec[h] = acc = rand() < acc_prob
            if !acc
                # Revert the change
                gammas_star[h] = gammas[h]
            end
        end
        
        # update the acceptance vector
        agammas_all[1:(H+2),iter] = agamma_vec
        
        # ----------------------------- Update Sigma_Gamma -----------------------------
        shape_param = 0.5 * H + 0.5
        # Optimized: compute sum of squared differences directly
        scale_param = 0.0
        @inbounds @simd for i in 1:(H+1)
            diff_val = gammas_star[i+1] - gammas_star[i]
            scale_param += diff_val * diff_val
        end
        scale_param *= 0.5
        sigma_gamma_star = sqrt(rand(InverseGamma(shape_param, scale_param)))
        
        # update the sequence
        sigmas_gamma_all[iter] = sigma_gamma_star
        
        # ----------------------------- Update Tau -----------------------------
        taus_ = [0 taus' tau][1,:]
        taus_star = copy(taus_)
        taus_star_replace = copy(taus_)
                
        if H > 0
             hc = rand(1:H)
             tau_hc_star = rand(Uniform(taus_star[hc],taus_star[hc+2]))
             taus_star_replace[hc+1] = tau_hc_star
             log_de = loglkh_cal(ws, X,Z,Delta,Y, 
                                        betas,
                                        tau,
                                        taus_star[2:(H+1)], 
                                        gammas_star,
                                        a,b,zetas,xis) +
                            log(taus_star[hc+2] - taus_star[hc+1]) + log(taus_star[hc+1] - taus_star[hc])
            
             log_num = loglkh_cal(ws, X,Z,Delta,Y, 
                                        betas,
                                        tau,
                                        taus_star_replace[2:(H+1)], 
                                        gammas_star,
                                        a,b,zetas,xis) + 
                            log(taus_star[hc+2] - tau_hc_star) + log(tau_hc_star - taus_star[hc])
                    
             aratio = exp(log_num - log_de)
             acc_prob = min(1, aratio)
             # accept or not
             acc = rand() < acc_prob
             # update the taus
             taus_star[hc+1] = acc * tau_hc_star + (1-acc) * taus_[hc+1] 
        end
        
        # # ------------ RJMCMC: Perform RJ-Update for (H, taus, gammas)------------
    
        # check the performance
        if H == 0
            r_HB_star = 1
            H_star = H + 1
        elseif H == Hmax
            r_HB_star = 0
            H_star = H - 1
        else
            r_HB_star = r_HB
            H_star = H + 2*Int(rand(Bernoulli(0.5))) - 1
        end
       
        if RJMCMC_indicator
            if H_star > H
                # ------------ codes for H_star > H (generate a new knot) ------------
            
                # sample tau from the unselected observed time
                tau_star = rand(Uniform(0,tau))
                
                # merge new tau into a new list 
                taus_star_add = sort([taus_star; tau_star])
                
                # propose a new gamma
                h = sum(taus_star .< tau_star) 
                Ah = (gammas_star[h+1] - gammas_star[h]) / (taus_star[h+1] - taus_star[h])
                U = rand()
                gamma_star = gammas_star[h] + (tau_star - taus_star[h]) * (Ah - (taus_star[h+1] - tau_star) 
                    / (taus_star[h+1] - taus_star[h]) * log((1-U)/U))
                # add the new gamma into the list
                gammas_star_add = [gammas_star[1:h]; gamma_star; gammas_star[(h+1):(H_star+1)]]
                
                # acceptance rate
                log_a_BM = loglkh_cal(ws, X,Z,Delta,Y, 
                                        betas,
                                        tau,
                                        taus_star_add[2:(H_star+1)], 
                                        gammas_star_add,
                                        a,b,zetas,xis) - 
                           loglkh_cal(ws, X,Z,Delta,Y, 
                                        betas,
                                        tau,
                                        taus_star[2:(H+1)], 
                                        gammas_star,
                                        a,b,zetas,xis) + 
                       log(2*H+3) + log(2*H+2) + log(tau_star-taus_star[h]) + log(taus_star[h+1]-tau_star) + 
                       logpdf_normal(gamma_star, gammas_star[h], sigma_gamma_star) +
                       logpdf_normal(gammas_star[h+1], gamma_star, sigma_gamma_star) -
                       2*log(tau) - log(taus_star[h+1] - taus_star[h]) - 
                       logpdf_normal(gammas_star[h+1], gammas_star[h], sigma_gamma_star) +
                       log((1-r_HB)/(H+1)) - log(r_HB_star/tau) - log(U*(1-U)) + log(mu_H) - log(H+1)
                
                acc_BM_prob = min(1, exp(log_a_BM))
                
                # accept or not for the Metropolis-Hastings' Move
                acc_MHM = rand() < acc_BM_prob
                
                # update the coefficients
                if acc_MHM
                    H_all[iter] = H_star
                    aH_all[iter] = 1
                    taus_all[1:H_star,iter] = taus_star_add[2:(H_star+1)]
                    gammas_all[1:(H_star+2),iter] = gammas_star_add
            
                    H = H_star
                    taus = taus_star_add[2:(H_star+1)]
                    gammas = gammas_star_add
                    sigma_gamma = sigma_gamma_star
                else
                    H_all[iter] = H
                    aH_all[iter] = 0
                    taus_all[1:H,iter] = taus_star[2:(H+1)]
                    gammas_all[1:(H+2),iter] = gammas_star
            
                    taus = taus_star[2:(H+1)]
                    gammas = gammas_star
                    sigma_gamma = sigma_gamma_star
                end
                    
            else 
                # ------------ codes for H_star < H (remove a new knot) ------------
            
                # select a random value
                h = rand([1:1:H;])
                
                # remove an element from taus
                taus_star_rm = copy(taus_star)
                taus_star_rm = deleteat!(taus_star_rm, h+1)
                
                # remove an element from gammas
                gammas_star_rm = copy(gammas_star)
                gammas_star_rm = deleteat!(gammas_star_rm, h+1)

				# update the calculation of U 1206
				A_hp1 = (gammas_star[h+2]-gammas_star[h+1])/(taus_star[h+2]-taus_star[h+1])
				A_h = (gammas_star[h+1]-gammas_star[h])/(taus_star[h+1]-taus_star[h])
				U = exp(A_h)/(exp(A_h)+exp(A_hp1))
                
                # logarithmic 
                log_a_DM = loglkh_cal(ws, X,Z,Delta,Y, 
                                    betas,
                                    tau,
                                    taus_star_rm[2:(H_star+1)], 
                                    gammas_star_rm,
                                    a,b,zetas,xis) - 
                        loglkh_cal(ws, X,Z,Delta,Y, 
                                    betas,
                                    tau,
                                    taus_star[2:(H+1)], 
                                    gammas_star,
                                    a,b,zetas,xis) + 
                       2*log(tau) + log(taus_star_rm[h+1]-taus_star_rm[h]) + 
                       logpdf_normal(gammas_star_rm[h+1], gammas_star_rm[h], sigma_gamma_star) - 
                       log(2*H+1) - log(2*H) - 
                       log(taus_star[h+2]-taus_star[h+1]) - log(taus_star[h+1]-taus_star[h]) - 
                       logpdf_normal(gammas_star[h+2], gammas_star[h+1], sigma_gamma_star) -
                       logpdf_normal(gammas_star[h+1], gammas_star[h], sigma_gamma_star) +
                       log(r_HB/tau) - log((1-r_HB_star)/H) + log(U*(1-U)) + log(H) - log(mu_H)
        
                acc_DM_prob = min(1, exp(log_a_DM))
                
                # accept or not for the Metropolis-Hastings' Move
                acc_MHM = rand() < acc_DM_prob
            
                # update the coefficients
                if acc_MHM
                    H_all[iter] = H_star
                    aH_all[iter] = 1
                    taus_all[1:H_star,iter] = taus_star_rm[2:(H_star+1)]
                    gammas_all[1:(H_star+2),iter] = gammas_star_rm
            
                    H = H_star
                    taus = taus_star_rm[2:(H_star+1)]
                    gammas = gammas_star_rm
                    sigma_gamma = sigma_gamma_star
                else
                    H_all[iter] = H
                    aH_all[iter] = 0
                    taus_all[1:H,iter] = taus_star[2:(H+1)]
                    gammas_all[1:(H+2),iter] = gammas_star
            
                    taus = taus_star[2:(H+1)]
                    gammas = gammas_star
                    sigma_gamma = sigma_gamma_star
                end
            end
        else
            H_all[iter] = H
            aH_all[iter] = 0
            taus_all[1:H,iter] = taus_star[2:(H+1)]
            gammas_all[1:(H+2),iter] = gammas_star
    
            taus = taus_star[2:(H+1)]
            gammas = gammas_star
            sigma_gamma = sigma_gamma_star
        end
    
        # ----------------------------- Update Beta -----------------------------
        betas_star = copy(betas)
        abeta_vec  = zeros(size(betas)[1])
        
        # Pre-compute baseline components that don't change during beta updates
        mul!(ws.eta, X, betas_star)
        g_fun_est!(ws.g_values, Z, a, b, zetas, xis, ws.knots_g)
        loglambda_fun_est!(ws.loglambda_values, Y, tau, taus, gammas, ws.knots_h)
        Lambda_fun_est!(ws.Lambda_values, Y, tau, taus, gammas, ws.knots_h, ws.cumhaz_h)
        @inbounds @simd for i in eachindex(ws.eta)
            ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
        end
        
        for j in 1:dim_beta
            # compute the denominator (current state)
            log_de = compute_loglkh_optimized(Delta, ws.loglambda_values, ws.Xbeta_plus_g, ws.Lambda_values)
            
            # propose a new beta
            beta_j_new = rand(Uniform(betas[j]-c_beta, betas[j]+c_beta))
            
            # Incremental update: only change Xbeta for beta_j
            beta_diff = beta_j_new - betas_star[j]
            @inbounds @simd for i in eachindex(ws.eta)
                ws.eta[i] += X[i, j] * beta_diff
                ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
            end
            
            # compute the numerator (proposed state)
            log_num = compute_loglkh_optimized(Delta, ws.loglambda_values, ws.Xbeta_plus_g, ws.Lambda_values)
            
            # acceptance ratio
            aratio   = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            # accept or not
            abeta_vec[j] = acc = rand() < acc_prob
            if !acc
                # Revert the change
                @inbounds @simd for i in eachindex(ws.eta)
                    ws.eta[i] -= X[i, j] * beta_diff
                    ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
                end
                betas_star[j] = betas[j]
            else
                betas_star[j] = beta_j_new
            end
        end
        
        # update the coefficients
        betas_all[:,iter] = betas_star
        abetas_all[:,iter] = abeta_vec
        betas = betas_star
    
        # ----------------------------- Update Xi ----------------------------- 
        xis_star = copy(xis)
        axi_vec  = zeros(size(xis)[1])
        sigma_xi = sigmas_xi_all[iter-1]
        
        # Pre-compute baseline components (Xbeta doesn't change during xi updates)
        mul!(ws.eta, X, betas)
        g_fun_est!(ws.g_values, Z, a, b, zetas, xis_star, ws.knots_g)
        loglambda_fun_est!(ws.loglambda_values, Y, tau, taus, gammas, ws.knots_h)
        Lambda_fun_est!(ws.Lambda_values, Y, tau, taus, gammas, ws.knots_h, ws.cumhaz_h)
        @inbounds @simd for i in eachindex(ws.eta)
            ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
        end
    
        for k in 2:(K+2)        
            # Save current state before proposing (needed for rejection case)
            # This is the value at the start of this iteration, which may have been
            # updated by previous accepted proposals in this loop
            xi_k_current = xis_star[k]
            
            # compute the denominator - use optimized logpdf_normal
            log_prob_de = 0.0
            @inbounds for kk in 2:(K+2)
                log_prob_de += logpdf_normal(xis_star[kk], xis_star[kk-1], sigma_xi)
            end
    
            log_de = log_prob_de + compute_loglkh_optimized(Delta, ws.loglambda_values, ws.Xbeta_plus_g, ws.Lambda_values)
            
            # compute the numerator
            # propose a new xi - use xis[k] (original state) to match model1229.jl exactly
            xi_k_new = rand(Uniform(xis[k]-c_xi, xis[k]+c_xi))
            xis_star[k] = xi_k_new
            
            log_prob_num = 0.0
            @inbounds for kk in 2:(K+2)
                log_prob_num += logpdf_normal(xis_star[kk], xis_star[kk-1], sigma_xi)
            end
            
            # Only recompute g_values (xi changed)
            g_fun_est!(ws.g_values, Z, a, b, zetas, xis_star, ws.knots_g)
            @inbounds @simd for i in eachindex(ws.eta)
                ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
            end
            
            log_num = log_prob_num + compute_loglkh_optimized(Delta, ws.loglambda_values, ws.Xbeta_plus_g, ws.Lambda_values)
            
            # acceptance ratio
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            # accept or not
            axi_vec[k] = acc = rand() < acc_prob
            if !acc
                # Revert the change - restore to saved current state
                xis_star[k] = xi_k_current
                g_fun_est!(ws.g_values, Z, a, b, zetas, xis_star, ws.knots_g)
                @inbounds for i in eachindex(ws.eta)
                    ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
                end
            end
        end
        
        # update the acceptance vector
        axis_all[1:(K+2),iter] = axi_vec
        
        # ----------------------------- Update Sigma_Xi -----------------------------
        shape_param = 0.5 * K + 0.5
        # Optimized: compute sum of squared differences directly
        scale_param = 0.0
        @inbounds @simd for i in 1:(K+1)
            diff_val = xis_star[i+1] - xis_star[i]
            scale_param += diff_val * diff_val
        end
        scale_param *= 0.5
        sigma_xi_star = sqrt(rand(InverseGamma(shape_param, scale_param)))
        
        # update the sequence
        sigmas_xi_all[iter] = sigma_xi_star
        
        # ----------------------------- Update Zetas -----------------------------
        zetas_ = [a zetas' b][1,:]
        zetas_star = copy(zetas_)
        zetas_star_replace = copy(zetas_)
		
        if K > 0
             kc = rand(1:K)
             zeta_kc_star = rand(Uniform(zetas_star[kc],zetas_star[kc+2]))
             zetas_star_replace[kc+1] = zeta_kc_star
             log_de = loglkh_cal(ws, X,Z,Delta,Y, 
								betas,
								tau,
								taus, 
								gammas,
								a,b,
								zetas_star[2:(K+1)],
								xis_star) +
                            log(zetas_star[kc+2] - zetas_star[kc+1]) + log(zetas_star[kc+1] - zetas_star[kc])
            
             log_num = loglkh_cal(ws, X,Z,Delta,Y, 
                                        betas,
                                        tau,
                                        taus, 
                                        gammas,
                                        a,b,
										zetas_star_replace[2:(K+1)],
										xis_star) + 
                            log(zetas_star[kc+2] - zeta_kc_star) + log(zeta_kc_star - zetas_star[kc])
                    
             aratio = exp(log_num - log_de)
             acc_prob = min(1, aratio)
             # accept or not
             acc = rand() < acc_prob
             # update the zetas
             zetas_star[kc+1] = acc * zeta_kc_star + (1-acc) * zetas_[kc+1] 
        end
    
        # ------------ RJMCMC: Perform RJ-Update for (K, zetas, xis)------------
      
        # check the performance
         if K == 0
            r_KB_star = 1
            K_star = K + 1
        elseif K == Kmax
            r_KB_star = 0
            K_star = K - 1
        else
            r_KB_star = r_KB
            K_star = K + 2*Int(rand(Bernoulli(0.5))) - 1
        end
    
        if RJMCMC_indicator
            if K_star > K
                # ------------ codes for K_star > K (generate a new knot) ------------
            
                # sample tau from the unselected observed Z variable
                #zeta_star = rand(setdiff(Set(zetas_can), Set(zetas_star)))
                zeta_star = rand(Uniform(a,b))
                
                # number of data points
                #D = size(Z)[1]
                
                # merge new tau into a new list 
                zetas_star_add = sort([zetas_star; zeta_star])
                
                # propose a new xi
                k = sum(zetas_star .< zeta_star) 
                Ak = (xis_star[k+1] - xis_star[k]) / (zetas_star[k+1] - zetas_star[k])
                U = rand()
                xi_star = xis_star[k] + (zeta_star - zetas_star[k]) * (Ak - (zetas_star[k+1] - zeta_star) 
                          / (zetas_star[k+1] - zetas_star[k]) * log((1-U)/U))
                # add the new xi into the list
                xis_star_add = [xis_star[1:k]; xi_star; xis_star[(k+1):(K_star+1)]]
                
                # acceptance rate
                log_a_BM = loglkh_cal(ws, X,Z,Delta,Y, 
                                     betas,
                                     tau,taus,gammas,
                                     a,b,zetas_star_add[2:(K_star+1)],
                                     xis_star_add) - 
                       loglkh_cal(ws, X,Z,Delta,Y, 
                                     betas,
                                     tau,taus,gammas,
                                     a,b,zetas_star[2:(K+1)],
                                     xis_star) + 
                       log(2*K+3) + log(2*K+2) + log(zeta_star-zetas_star[k]) + log(zetas_star[k+1]-zeta_star) + 
                       logpdf_normal(xi_star, xis_star[k], sigma_xi_star) +
                       logpdf_normal(xis_star[k+1], xi_star, sigma_xi_star) - 
                       2*log(b-a) - log(zetas_star[k+1] - zetas_star[k]) - 
                       logpdf_normal(xis_star[k+1], xis_star[k], sigma_xi_star) +
                       log((1-r_KB)/(K+1)) - log(r_KB_star/(b-a)) - log(U*(1-U)) + log(mu_K) - log(K+1)
                
                acc_BM_prob = min(1, exp(log_a_BM))
                
                # accept or not for the Metropolis-Hastings' Move
                acc_MHM = rand() < acc_BM_prob
                # acc_MHM = false
            
                # update the coefficients
                if acc_MHM
                    K_all[iter] = K_star
                    aK_all[iter] = 1
                    zetas_all[1:K_star,iter] = zetas_star_add[2:(K_star+1)]
                    xis_all[1:(K_star+2),iter] = xis_star_add
            
                    K = K_star
                    zetas = zetas_star_add[2:(K_star+1)]
                    xis = xis_star_add
                    sigma_xi = sigma_xi_star
                else
                    K_all[iter] = K
                    aK_all[iter] = 0
                    zetas_all[1:K,iter] = zetas_star[2:(K+1)]
                    xis_all[1:(K+2),iter] = xis_star
            
                    zetas = zetas_star[2:(K+1)]
                    xis = xis_star
                    sigma_xi = sigma_xi_star
                end
                    
            else 
                # ------------ codes for K_star < K (remove a new knot) ------------
                
                # generate the random results
                # select a random value
                k = rand([1:1:K;])
                
                # remove an element from taus
                zetas_star_rm = copy(zetas_star)
                zetas_star_rm = deleteat!(zetas_star_rm, k+1)
                
                # remove an element from gammas
                xis_star_rm = copy(xis_star)
                xis_star_rm = deleteat!(xis_star_rm, k+1)

                # update the calculation of U 1206
				A_hp1 = (xis_star[k+2]-xis_star[k+1])/(zetas_star[k+2]-zetas_star[k+1])
				A_h = (xis_star[k+1]-xis_star[k])/(zetas_star[k+1]-zetas_star[k])
				U = exp(A_h)/(exp(A_h)+exp(A_hp1))
                
                # logarithmic 
                log_a_DM = loglkh_cal(ws, X,Z,Delta,Y, 
                                     betas,
                                     tau,taus,gammas,
                                     a,b,zetas_star_rm[2:(K_star+1)],
                                     xis_star_rm) - 
                           loglkh_cal(ws, X,Z,Delta,Y, 
                                     betas,
                                     tau,taus,gammas,
                                     a,b,
                                     zetas_star[2:(K+1)],
                                     xis_star)  +
                           2*log(b-a) + log(zetas_star_rm[k+1]-zetas_star_rm[k]) + 
                           logpdf_normal(xis_star_rm[k+1], xis_star_rm[k], sigma_xi_star) - 
                           log(2*K+1) - log(2*K) - 
                           log(zetas_star[k+2]-zetas_star[k+1]) - log(zetas_star[k+1]-zetas_star[k]) - 
                           logpdf_normal(xis_star[k+2], xis_star[k+1], sigma_xi_star) -
                           logpdf_normal(xis_star[k+1], xis_star[k], sigma_xi_star) +
                           log(r_KB/(b-a)) - log((1-r_KB_star)/K) + log(U*(1-U)) + log(K) - log(mu_K)
                
                acc_DM_prob = min(1, exp(log_a_DM))
                
                # accept or not for the Metropolis-Hastings' Move
                acc_MHM = rand() < acc_DM_prob
                # acc_MHM = false
            
                # update the coefficients
                if acc_MHM
                    K_all[iter] = K_star
                    aK_all[iter] = 1
                        if K_star == 0
                            zetas_all[1:K_star,iter] = zeros(K_star)
                        else
                            zetas_all[1:K_star,iter] = zetas_star_rm[2:(K_star+1)]
                        end
                    xis_all[1:(K_star+2),iter] = xis_star_rm
            
                    K = K_star
                    zetas = zetas_star_rm[2:(K_star+1)]
                    xis = xis_star_rm
                    sigma_xi = sigma_xi_star
                else
                    K_all[iter] = K
                    aK_all[iter] = 0
                    zetas_all[1:K,iter] = zetas_star[2:(K+1)]
                    xis_all[1:(K+2),iter] = xis_star
            
                    zetas = zetas_star[2:(K+1)]
                    xis = xis_star
                    sigma_xi = sigma_xi_star
                end
            end
        else
            K_all[iter] = K
            aK_all[iter] = 0
            zetas_all[1:K,iter] = zetas_star[2:(K+1)]
            xis_all[1:(K+2),iter] = xis_star
    
            zetas = zetas_star[2:(K+1)]
            xis = xis_star
            sigma_xi = sigma_xi_star
        end
            
        if iter % n_report == 0
    
            if Adapt_C
                avg_gammas_aratio = mean(sum(agammas_all[:,(n_report*c_cnt+1):(n_report*(c_cnt+1))], dims=2) ./ n_report)
                avg_xis_aratio = mean(sum(axis_all[:,(n_report*c_cnt+1):(n_report*(c_cnt+1))],dims=2) ./ n_report)
                avg_betas_aratio = mean(sum(abetas_all[:,(n_report*c_cnt+1):(n_report*(c_cnt+1))],dims=2) ./ n_report) 
                
                if avg_gammas_aratio > AHigh
                    c_gamma = min(c_gamma*2, 2)
                end
                
                if avg_gammas_aratio < ALow
                    c_gamma = max(c_gamma/2, 0.0625)
                end
                
                if avg_xis_aratio > AHigh
                    c_xi = min(c_xi*2, 2)
                end
                
                if avg_xis_aratio < ALow
                    c_xi = max(c_xi/2, 0.0625)
                end
                
                if avg_betas_aratio > AHigh
                    c_beta = min(c_beta*2, 2)
                end
                
                if avg_betas_aratio < ALow
                    c_beta = max(c_beta/2, 0.0625)
                end
                
                c_cnt += 1
                
            else
                print("Iteration ", iter, " \n")
            end 
        end
        if show_progress
            next!(prog)
        end
    end

    results = Dict(
        "H"=>H_all,
        "taus"=>taus_all,
        "gammas"=>gammas_all,
        "sigmas_gamma"=>sigmas_gamma_all,
        "K"=>K_all,
        "zetas"=>zetas_all,
        "xis"=>xis_all,
        "sigmas_xi"=>sigmas_xi_all,
        "betas"=>betas_all,
        "tau"=> tau,
        "Tmax" => tau,
        "a"=> a,      
        "b"=> b,
        "ns" => NS,
        "burn_in" => BI
    )

    return results
end

##################################################################################################################
# RJMCMC with Dirichlet-Gamma Prior for Knot Locations
# Based on Supplementary Material: Dirichlet Priors for Knot Locations
##################################################################################################################

# Compute log Dirichlet prior for knot locations
function log_dirichlet_prior_tau(taus_, alpha_tau, tau)
    H = length(taus_) - 2  # taus_ includes 0 and tau as boundaries
    if H < 0
        return 0.0
    end
    # Compute interval lengths
    deltas = diff(taus_)
    # Log prior: log Γ((H+1)α) - (H+1)log Γ(α) + (α-1) * Σ log(Δ_h)
    log_prior = loggamma((H+1) * alpha_tau) - (H+1) * loggamma(alpha_tau) + (alpha_tau - 1) * sum(log.(deltas))
    return log_prior
end

function log_dirichlet_prior_zeta(zetas_, alpha_zeta, a, b)
    K = length(zetas_) - 2  # zetas_ includes a and b as boundaries
    if K < 0
        return 0.0
    end
    # Compute interval lengths
    deltas = diff(zetas_)
    # Log prior: log Γ((K+1)α) - (K+1)log Γ(α) + (α-1) * Σ log(Δ_k)
    log_prior = loggamma((K+1) * alpha_zeta) - (K+1) * loggamma(alpha_zeta) + (alpha_zeta - 1) * sum(log.(deltas))
    return log_prior
end

# Reversible-jump sampler with Dirichlet-Gamma prior for knot locations
function RJMCMC_Nonlinear_Dirichlet(
        X, Z, Delta, Y, a, b,
        random_seed;
        ns::Int=DEFAULT_NS,
        burn_in::Int=DEFAULT_BURN_IN,
        RJMCMC_indicator::Bool=true,
        Adapt_C::Bool=true,
        Hmax::Int=DEFAULT_HMAX,
        Kmax::Int=DEFAULT_KMAX,
        # Dirichlet-Gamma hyperparameters
        a_tau::Float64=1.0,    # Gamma shape for alpha_tau
        b_tau::Float64=1.0,    # Gamma rate for alpha_tau
        a_zeta::Float64=1.0,   # Gamma shape for alpha_zeta
        b_zeta::Float64=1.0,   # Gamma rate for alpha_zeta
        show_progress::Bool=true
     )

    NS = ns
    BI = burn_in
    # ---------------------------- Hyper-Parameters ----------------------------
    dim_beta = size(X,2)
    ws = LikelihoodWorkspace(length(Y), Hmax, Kmax)
    
    # obtain the observed time
    obs_time = (Y .* Delta) 
    obs_time = obs_time[obs_time .> 0]
    tau  = maximum(obs_time)
    
    # prior parameter for number of probability (matching NonLinear1)
    mu_H = mu_K = 1

    # initial H and K
    H = 0
    K = 0
    
    # Initial Dirichlet concentration parameters (sampled from Gamma prior)
    alpha_tau = rand(Gamma(a_tau, 1/b_tau))
    alpha_zeta = rand(Gamma(a_zeta, 1/b_zeta))
    
    # coefficients for MH proposals
    c_gamma = 1
    c_xi = 1
    c_beta = 1
    c_cnt = 0
    c_alpha = 0.5  # for alpha updates
    AHigh = 0.4
    ALow = 0.2
    
    n_report = 250
    
    # birth probability
    r_HB = 0.5
    r_KB = 0.5
    
    # ---------------------------- Storage Variables ----------------------------
    betas_all = zeros(dim_beta, NS)
    
    H_all = zeros(NS)
    taus_all = zeros(Hmax, NS)
    gammas_all = zeros(Hmax+2, NS)
    
    K_all = zeros(NS)
    zetas_all = zeros(Kmax, NS)
    xis_all = zeros(Kmax+2, NS)
    
    sigmas_gamma_all = zeros(NS)
    sigmas_xi_all = zeros(NS)
    
    # Store Dirichlet concentration parameters
    alpha_tau_all = zeros(NS)
    alpha_zeta_all = zeros(NS)
    
    # Acceptance variables
    abetas_all = zeros(dim_beta, NS)
    aH_all = zeros(NS)
    agammas_all = zeros(Hmax+2, NS)
    aK_all = zeros(NS)
    axis_all = zeros(Kmax+2, NS)
    
    # ------------------ Prior Specification ------------------ 
    Random.seed!(random_seed)
    
    # Create candidate sets (matching NonLinear1 exactly)
    Hcan = 20  # Candidate set size for baseline hazard knots
    Kcan = 20  # Candidate set size for g(z) knots
    taus_can = LinRange(0, tau, Hcan+2)[2:(Hcan+1)]
    zetas_can = LinRange(a, b, Kcan+2)[2:(Kcan+1)]
    
    # Initialize taus using candidate set method (matching NonLinear1)
    if H > 0
        taus = sort(sample(taus_can, H, replace=false))
    else
        taus = Float64[]  # Empty when H=0
    end
    taus_ = [0 taus' tau][1,:]
    
    gammas = zeros(H+2)
    gammas[1] = rand(Normal(0,5))
    for i in 2:(H+2)
        gammas[i] = rand(Normal(gammas[i-1],1))
    end  
    
    # Initialize zetas using candidate set method (matching NonLinear1)
    if K > 0
        zetas = sort(sample(zetas_can, K, replace=false))
    else
        zetas = Float64[]  # Empty when K=0
    end
    zetas_ = [a zetas' b][1,:]
    
    xis = zeros(K+2)
    xis[1] = 0
    for i in 2:(K+2)
        xis[i] = rand(Normal(xis[i-1],1))
    end
    
    betas = zeros(dim_beta)
    
    # Store initial values
    H_all[1] = H
    taus_all[1:H,1] = taus
    gammas_all[1:(H+2),1] = gammas
    K_all[1] = K
    zetas_all[1:K,1] = zetas
    xis_all[1:(K+2),1] = xis
    betas_all[:,1] = betas
    sigmas_gamma_all[1] = 1
    sigmas_xi_all[1] = 1
    alpha_tau_all[1] = alpha_tau
    alpha_zeta_all[1] = alpha_zeta
    
    aH_all[1] = 1
    agammas_all[1:(H+2),1] .= 1
    aK_all[1] = 1
    axis_all[1:(K+2),1] .= 1
    abetas_all[:,1] .= 1

    Random.seed!(random_seed)

    # ---------------------------- MCMC Loop ----------------------------
    # Only create progress bar if explicitly enabled
    if show_progress
        prog = Progress(NS - 1; desc="MCMC")
    end
    for iter in 2:NS

        # ----------------------------- Update Gamma ----------------------------- 
        gammas_star = copy(gammas)
        agamma_vec = zeros(size(gammas)[1])
        sigma_gamma = sigmas_gamma_all[iter-1]
        
        # Pre-compute components that don't change during gamma updates
        mul!(ws.eta, X, betas)
        g_fun_est!(ws.g_values, Z, a, b, zetas, xis, ws.knots_g)
        
        for h in 1:(H+2)        
            log_prob_de = logpdf_normal(gammas_star[1], 0.0, 5.0)
            @inbounds for hh in 2:(H+2)
                log_prob_de += logpdf_normal(gammas_star[hh], gammas_star[hh-1], sigma_gamma)
            end
            
            # Compute loglambda and Lambda for current gammas
            loglambda_fun_est!(ws.loglambda_values, Y, tau, taus, gammas_star, ws.knots_h)
            Lambda_fun_est!(ws.Lambda_values, Y, tau, taus, gammas_star, ws.knots_h, ws.cumhaz_h)
            @inbounds @simd for i in eachindex(ws.eta)
                ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
            end
            log_de = log_prob_de + compute_loglkh_optimized(Delta, ws.loglambda_values, ws.Xbeta_plus_g, ws.Lambda_values)
            
            # propose a new gamma
            gamma_h_new = rand(Uniform(gammas[h]-c_gamma, gammas[h]+c_gamma))
            gammas_star[h] = gamma_h_new
            
            log_prob_num = logpdf_normal(gammas_star[1], 0.0, 5.0)
            @inbounds for hh in 2:(H+2)
                log_prob_num += logpdf_normal(gammas_star[hh], gammas_star[hh-1], sigma_gamma)
            end
            
            # Recompute loglambda and Lambda for proposed gammas
            loglambda_fun_est!(ws.loglambda_values, Y, tau, taus, gammas_star, ws.knots_h)
            Lambda_fun_est!(ws.Lambda_values, Y, tau, taus, gammas_star, ws.knots_h, ws.cumhaz_h)
            @inbounds @simd for i in eachindex(ws.eta)
                ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
            end
            log_num = log_prob_num + compute_loglkh_optimized(Delta, ws.loglambda_values, ws.Xbeta_plus_g, ws.Lambda_values)
            
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            agamma_vec[h] = acc = rand() < acc_prob
            if !acc
                # Revert the change
                gammas_star[h] = gammas[h]
            end
        end
        
        agammas_all[1:(H+2),iter] = agamma_vec
        
        # ----------------------------- Update Sigma_Gamma -----------------------------
        shape_param = 0.5 * H + 0.5
        # Optimized: compute sum of squared differences directly
        scale_param = 0.0
        @inbounds @simd for i in 1:(H+1)
            diff_val = gammas_star[i+1] - gammas_star[i]
            scale_param += diff_val * diff_val
        end
        scale_param *= 0.5
        sigma_gamma_star = sqrt(rand(InverseGamma(shape_param, scale_param)))
        sigmas_gamma_all[iter] = sigma_gamma_star
        
        # ----------------------------- Update Tau (with Dirichlet prior) -----------------------------
        taus_ = [0 taus' tau][1,:]
        taus_star = copy(taus_)
        taus_star_replace = copy(taus_)
                
        if H > 0
            hc = rand(1:H)
            tau_hc_star = rand(Uniform(taus_star[hc], taus_star[hc+2]))
            taus_star_replace[hc+1] = tau_hc_star
            
            # Log-likelihood ratio
            log_lik_de = loglkh_cal(ws, X,Z,Delta,Y, betas, tau, taus_star[2:(H+1)], gammas_star, a,b,zetas,xis)
            log_lik_num = loglkh_cal(ws, X,Z,Delta,Y, betas, tau, taus_star_replace[2:(H+1)], gammas_star, a,b,zetas,xis)
            
            # Dirichlet prior ratio (Eq. 3 in supplement)
            log_prior_de = (alpha_tau - 1) * (log(taus_star[hc+2] - taus_star[hc+1]) + log(taus_star[hc+1] - taus_star[hc]))
            log_prior_num = (alpha_tau - 1) * (log(taus_star[hc+2] - tau_hc_star) + log(tau_hc_star - taus_star[hc]))
            
            log_ratio = (log_lik_num - log_lik_de) + (log_prior_num - log_prior_de)
            
            aratio = exp(log_ratio)
            acc_prob = min(1, aratio)
            acc = rand() < acc_prob
            taus_star[hc+1] = acc * tau_hc_star + (1-acc) * taus_[hc+1]
        end
        
        # ----------------------------- RJMCMC for H (with Dirichlet prior) -----------------------------
        if H == 0
            r_HB_star = 1
            H_star = H + 1
        elseif H == Hmax
            r_HB_star = 0
            H_star = H - 1
        else
            r_HB_star = r_HB
            H_star = H + 2*Int(rand(Bernoulli(0.5))) - 1
        end
       
        if RJMCMC_indicator
            if H_star > H
                # ------------ Birth Move (Eq. 4 in supplement) ------------
                tau_star_new = rand(Uniform(0, tau))
                taus_star_add = sort([taus_star; tau_star_new])
                
                h = sum(taus_star .< tau_star_new)
                Ah = (gammas_star[h+1] - gammas_star[h]) / (taus_star[h+1] - taus_star[h])
                U = rand()
                gamma_star = gammas_star[h] + (tau_star_new - taus_star[h]) * (Ah - (taus_star[h+1] - tau_star_new) / (taus_star[h+1] - taus_star[h]) * log((1-U)/U))
                gammas_star_add = [gammas_star[1:h]; gamma_star; gammas_star[(h+1):(H_star+1)]]
                
                # Dirichlet prior ratio for birth (Eq. 4)
                Delta_j_old = taus_star[h+1] - taus_star[h]
                Delta_j_new = tau_star_new - taus_star[h]
                Delta_jp1_new = taus_star[h+1] - tau_star_new
                
                log_prior_ratio = loggamma((H+2) * alpha_tau) - loggamma((H+1) * alpha_tau) - loggamma(alpha_tau) +
                                  (alpha_tau - 1) * (log(Delta_j_new) + log(Delta_jp1_new) - log(Delta_j_old))
                
                log_a_BM = loglkh_cal(ws, X,Z,Delta,Y, betas, tau, taus_star_add[2:(H_star+1)], gammas_star_add, a,b,zetas,xis) - 
                           loglkh_cal(ws, X,Z,Delta,Y, betas, tau, taus_star[2:(H+1)], gammas_star, a,b,zetas,xis) +
                           log_prior_ratio +
                           logpdf_normal(gamma_star, gammas_star[h], sigma_gamma_star) +
                           logpdf_normal(gammas_star[h+1], gamma_star, sigma_gamma_star) -
                           logpdf_normal(gammas_star[h+1], gammas_star[h], sigma_gamma_star) +
                           log((1-r_HB)/(H+1)) - log(r_HB_star/tau) - log(U*(1-U)) + log(mu_H) - log(H+1)
                
                acc_BM_prob = min(1, exp(log_a_BM))
                acc_MHM = rand() < acc_BM_prob
                
                if acc_MHM
                    H_all[iter] = H_star
                    aH_all[iter] = 1
                    taus_all[1:H_star,iter] = taus_star_add[2:(H_star+1)]
                    gammas_all[1:(H_star+2),iter] = gammas_star_add
                    H = H_star
                    taus = taus_star_add[2:(H_star+1)]
                    gammas = gammas_star_add
                    sigma_gamma = sigma_gamma_star
                else
                    H_all[iter] = H
                    aH_all[iter] = 0
                    taus_all[1:H,iter] = taus_star[2:(H+1)]
                    gammas_all[1:(H+2),iter] = gammas_star
                    taus = taus_star[2:(H+1)]
                    gammas = gammas_star
                    sigma_gamma = sigma_gamma_star
                end
                    
            else 
                # ------------ Death Move (Eq. 5 in supplement) ------------
                h = rand(1:H)
                taus_star_rm = deleteat!(copy(taus_star), h+1)
                gammas_star_rm = deleteat!(copy(gammas_star), h+1)

                A_hp1 = (gammas_star[h+2]-gammas_star[h+1])/(taus_star[h+2]-taus_star[h+1])
                A_h = (gammas_star[h+1]-gammas_star[h])/(taus_star[h+1]-taus_star[h])
                U = exp(A_h)/(exp(A_h)+exp(A_hp1))
                
                # Dirichlet prior ratio for death (Eq. 5)
                Delta_j_old = taus_star[h+1] - taus_star[h]
                Delta_jp1_old = taus_star[h+2] - taus_star[h+1]
                Delta_j_new = taus_star_rm[h+1] - taus_star_rm[h]
                
                log_prior_ratio = loggamma(H * alpha_tau) + loggamma(alpha_tau) - loggamma((H+1) * alpha_tau) +
                                  (alpha_tau - 1) * (log(Delta_j_new) - log(Delta_j_old) - log(Delta_jp1_old))
                
                log_a_DM = loglkh_cal(ws, X,Z,Delta,Y, betas, tau, taus_star_rm[2:(H_star+1)], gammas_star_rm, a,b,zetas,xis) - 
                           loglkh_cal(ws, X,Z,Delta,Y, betas, tau, taus_star[2:(H+1)], gammas_star, a,b,zetas,xis) +
                           log_prior_ratio +
                           logpdf_normal(gammas_star_rm[h+1], gammas_star_rm[h], sigma_gamma_star) - 
                           logpdf_normal(gammas_star[h+2], gammas_star[h+1], sigma_gamma_star) -
                           logpdf_normal(gammas_star[h+1], gammas_star[h], sigma_gamma_star) +
                           log(r_HB/tau) - log((1-r_HB_star)/H) + log(U*(1-U)) + log(H) - log(mu_H)
        
                acc_DM_prob = min(1, exp(log_a_DM))
                acc_MHM = rand() < acc_DM_prob
            
                if acc_MHM
                    H_all[iter] = H_star
                    aH_all[iter] = 1
                    taus_all[1:H_star,iter] = taus_star_rm[2:(H_star+1)]
                    gammas_all[1:(H_star+2),iter] = gammas_star_rm
                    H = H_star
                    taus = taus_star_rm[2:(H_star+1)]
                    gammas = gammas_star_rm
                    sigma_gamma = sigma_gamma_star
                else
                    H_all[iter] = H
                    aH_all[iter] = 0
                    taus_all[1:H,iter] = taus_star[2:(H+1)]
                    gammas_all[1:(H+2),iter] = gammas_star
                    taus = taus_star[2:(H+1)]
                    gammas = gammas_star
                    sigma_gamma = sigma_gamma_star
                end
            end
        else
            H_all[iter] = H
            aH_all[iter] = 0
            taus_all[1:H,iter] = taus_star[2:(H+1)]
            gammas_all[1:(H+2),iter] = gammas_star
            taus = taus_star[2:(H+1)]
            gammas = gammas_star
            sigma_gamma = sigma_gamma_star
        end
    
        # ----------------------------- Update Beta -----------------------------
        betas_star = copy(betas)
        abeta_vec = zeros(size(betas)[1])
        
        # Pre-compute baseline components that don't change during beta updates
        mul!(ws.eta, X, betas_star)
        g_fun_est!(ws.g_values, Z, a, b, zetas, xis, ws.knots_g)
        loglambda_fun_est!(ws.loglambda_values, Y, tau, taus, gammas, ws.knots_h)
        Lambda_fun_est!(ws.Lambda_values, Y, tau, taus, gammas, ws.knots_h, ws.cumhaz_h)
        @inbounds @simd for i in eachindex(ws.eta)
            ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
        end
        
        for j in 1:dim_beta
            # compute the denominator (current state)
            log_de = compute_loglkh_optimized(Delta, ws.loglambda_values, ws.Xbeta_plus_g, ws.Lambda_values)
            
            # propose a new beta
            beta_j_new = rand(Uniform(betas[j]-c_beta, betas[j]+c_beta))
            
            # Incremental update: only change Xbeta for beta_j
            beta_diff = beta_j_new - betas_star[j]
            @inbounds @simd for i in eachindex(ws.eta)
                ws.eta[i] += X[i, j] * beta_diff
                ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
            end
            
            # compute the numerator (proposed state)
            log_num = compute_loglkh_optimized(Delta, ws.loglambda_values, ws.Xbeta_plus_g, ws.Lambda_values)
            
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            abeta_vec[j] = acc = rand() < acc_prob
            if !acc
                # Revert the change
                @inbounds @simd for i in eachindex(ws.eta)
                    ws.eta[i] -= X[i, j] * beta_diff
                    ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
                end
                betas_star[j] = betas[j]
            else
                betas_star[j] = beta_j_new
            end
        end
        
        betas_all[:,iter] = betas_star
        abetas_all[:,iter] = abeta_vec
        betas = betas_star
    
        # ----------------------------- Update Xi ----------------------------- 
        xis_star = copy(xis)
        axi_vec = zeros(size(xis)[1])
        sigma_xi = sigmas_xi_all[iter-1]
        
        # Pre-compute baseline components (Xbeta doesn't change during xi updates)
        mul!(ws.eta, X, betas)
        g_fun_est!(ws.g_values, Z, a, b, zetas, xis_star, ws.knots_g)
        loglambda_fun_est!(ws.loglambda_values, Y, tau, taus, gammas, ws.knots_h)
        Lambda_fun_est!(ws.Lambda_values, Y, tau, taus, gammas, ws.knots_h, ws.cumhaz_h)
        @inbounds @simd for i in eachindex(ws.eta)
            ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
        end
    
        for k in 2:(K+2)        
            # Save current state before proposing (needed for rejection case)
            # This is the value at the start of this iteration, which may have been
            # updated by previous accepted proposals in this loop
            xi_k_current = xis_star[k]
            
            log_prob_de = 0.0
            @inbounds for kk in 2:(K+2)
                log_prob_de += logpdf_normal(xis_star[kk], xis_star[kk-1], sigma_xi)
            end
    
            log_de = log_prob_de + compute_loglkh_optimized(Delta, ws.loglambda_values, ws.Xbeta_plus_g, ws.Lambda_values)
            
            # propose a new xi - use xis[k] (original state) to match NonLinear1 exactly
            xi_k_new = rand(Uniform(xis[k]-c_xi, xis[k]+c_xi))
            xis_star[k] = xi_k_new
            
            log_prob_num = 0.0
            @inbounds for kk in 2:(K+2)
                log_prob_num += logpdf_normal(xis_star[kk], xis_star[kk-1], sigma_xi)
            end
            
            # Only recompute g_values (xi changed)
            g_fun_est!(ws.g_values, Z, a, b, zetas, xis_star, ws.knots_g)
            @inbounds @simd for i in eachindex(ws.eta)
                ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
            end
            
            log_num = log_prob_num + compute_loglkh_optimized(Delta, ws.loglambda_values, ws.Xbeta_plus_g, ws.Lambda_values)
            
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            axi_vec[k] = acc = rand() < acc_prob
            if !acc
                # Revert the change - restore to saved current state
                xis_star[k] = xi_k_current
                g_fun_est!(ws.g_values, Z, a, b, zetas, xis_star, ws.knots_g)
                @inbounds for i in eachindex(ws.eta)
                    ws.Xbeta_plus_g[i] = ws.eta[i] + ws.g_values[i]
                end
            end
        end
        
        axis_all[1:(K+2),iter] = axi_vec
        
        # ----------------------------- Update Sigma_Xi -----------------------------
        shape_param = 0.5 * K + 0.5
        # Optimized: compute sum of squared differences directly
        scale_param = 0.0
        @inbounds @simd for i in 1:(K+1)
            diff_val = xis_star[i+1] - xis_star[i]
            scale_param += diff_val * diff_val
        end
        scale_param *= 0.5
        sigma_xi_star = sqrt(rand(InverseGamma(shape_param, scale_param)))
        sigmas_xi_all[iter] = sigma_xi_star
        
        # ----------------------------- Update Zeta (with Dirichlet prior) -----------------------------
        zetas_ = [a zetas' b][1,:]
        zetas_star = copy(zetas_)
        zetas_star_replace = copy(zetas_)
		
        if K > 0
            kc = rand(1:K)
            zeta_kc_star = rand(Uniform(zetas_star[kc], zetas_star[kc+2]))
            zetas_star_replace[kc+1] = zeta_kc_star
            
            # Log-likelihood ratio
            log_lik_de = loglkh_cal(ws, X,Z,Delta,Y, betas, tau, taus, gammas, a,b, zetas_star[2:(K+1)], xis_star)
            log_lik_num = loglkh_cal(ws, X,Z,Delta,Y, betas, tau, taus, gammas, a,b, zetas_star_replace[2:(K+1)], xis_star)
            
            # Dirichlet prior ratio
            log_prior_de = (alpha_zeta - 1) * (log(zetas_star[kc+2] - zetas_star[kc+1]) + log(zetas_star[kc+1] - zetas_star[kc]))
            log_prior_num = (alpha_zeta - 1) * (log(zetas_star[kc+2] - zeta_kc_star) + log(zeta_kc_star - zetas_star[kc]))
            
            log_ratio = (log_lik_num - log_lik_de) + (log_prior_num - log_prior_de)
            
            aratio = exp(log_ratio)
            acc_prob = min(1, aratio)
            acc = rand() < acc_prob
            zetas_star[kc+1] = acc * zeta_kc_star + (1-acc) * zetas_[kc+1]
        end
    
        # ----------------------------- RJMCMC for K (with Dirichlet prior) -----------------------------
        if K == 0
            r_KB_star = 1
            K_star = K + 1
        elseif K == Kmax
            r_KB_star = 0
            K_star = K - 1
        else
            r_KB_star = r_KB
            K_star = K + 2*Int(rand(Bernoulli(0.5))) - 1
        end
    
        if RJMCMC_indicator
            if K_star > K
                # ------------ Birth Move for zeta ------------
                zeta_star_new = rand(Uniform(a, b))
                zetas_star_add = sort([zetas_star; zeta_star_new])
                
                k = sum(zetas_star .< zeta_star_new)
                Ak = (xis_star[k+1] - xis_star[k]) / (zetas_star[k+1] - zetas_star[k])
                U = rand()
                xi_star = xis_star[k] + (zeta_star_new - zetas_star[k]) * (Ak - (zetas_star[k+1] - zeta_star_new) / (zetas_star[k+1] - zetas_star[k]) * log((1-U)/U))
                xis_star_add = [xis_star[1:k]; xi_star; xis_star[(k+1):(K_star+1)]]
                
                # Dirichlet prior ratio for birth
                Delta_k_old = zetas_star[k+1] - zetas_star[k]
                Delta_k_new = zeta_star_new - zetas_star[k]
                Delta_kp1_new = zetas_star[k+1] - zeta_star_new
                
                log_prior_ratio = loggamma((K+2) * alpha_zeta) - loggamma((K+1) * alpha_zeta) - loggamma(alpha_zeta) +
                                  (alpha_zeta - 1) * (log(Delta_k_new) + log(Delta_kp1_new) - log(Delta_k_old))
                
                log_a_BM = loglkh_cal(ws, X,Z,Delta,Y, betas, tau,taus,gammas, a,b, zetas_star_add[2:(K_star+1)], xis_star_add) - 
                           loglkh_cal(ws, X,Z,Delta,Y, betas, tau,taus,gammas, a,b, zetas_star[2:(K+1)], xis_star) +
                           log_prior_ratio +
                           logpdf_normal(xi_star, xis_star[k], sigma_xi_star) +
                           logpdf_normal(xis_star[k+1], xi_star, sigma_xi_star) - 
                           logpdf_normal(xis_star[k+1], xis_star[k], sigma_xi_star) +
                           log((1-r_KB)/(K+1)) - log(r_KB_star/(b-a)) - log(U*(1-U)) + log(mu_K) - log(K+1)
                
                acc_BM_prob = min(1, exp(log_a_BM))
                acc_MHM = rand() < acc_BM_prob
            
                if acc_MHM
                    K_all[iter] = K_star
                    aK_all[iter] = 1
                    zetas_all[1:K_star,iter] = zetas_star_add[2:(K_star+1)]
                    xis_all[1:(K_star+2),iter] = xis_star_add
                    K = K_star
                    zetas = zetas_star_add[2:(K_star+1)]
                    xis = xis_star_add
                    sigma_xi = sigma_xi_star
                else
                    K_all[iter] = K
                    aK_all[iter] = 0
                    zetas_all[1:K,iter] = zetas_star[2:(K+1)]
                    xis_all[1:(K+2),iter] = xis_star
                    zetas = zetas_star[2:(K+1)]
                    xis = xis_star
                    sigma_xi = sigma_xi_star
                end
                    
            else 
                # ------------ Death Move for zeta ------------
                k = rand(1:K)
                zetas_star_rm = deleteat!(copy(zetas_star), k+1)
                xis_star_rm = deleteat!(copy(xis_star), k+1)

                A_kp1 = (xis_star[k+2]-xis_star[k+1])/(zetas_star[k+2]-zetas_star[k+1])
                A_k = (xis_star[k+1]-xis_star[k])/(zetas_star[k+1]-zetas_star[k])
                U = exp(A_k)/(exp(A_k)+exp(A_kp1))
                
                # Dirichlet prior ratio for death
                Delta_k_old = zetas_star[k+1] - zetas_star[k]
                Delta_kp1_old = zetas_star[k+2] - zetas_star[k+1]
                Delta_k_new = zetas_star_rm[k+1] - zetas_star_rm[k]
                
                log_prior_ratio = loggamma(K * alpha_zeta) + loggamma(alpha_zeta) - loggamma((K+1) * alpha_zeta) +
                                  (alpha_zeta - 1) * (log(Delta_k_new) - log(Delta_k_old) - log(Delta_kp1_old))
                
                log_a_DM = loglkh_cal(ws, X,Z,Delta,Y, betas, tau,taus,gammas, a,b, zetas_star_rm[2:(K_star+1)], xis_star_rm) - 
                           loglkh_cal(ws, X,Z,Delta,Y, betas, tau,taus,gammas, a,b, zetas_star[2:(K+1)], xis_star) +
                           log_prior_ratio +
                           logpdf_normal(xis_star_rm[k+1], xis_star_rm[k], sigma_xi_star) - 
                           logpdf_normal(xis_star[k+2], xis_star[k+1], sigma_xi_star) -
                           logpdf_normal(xis_star[k+1], xis_star[k], sigma_xi_star) +
                           log(r_KB/(b-a)) - log((1-r_KB_star)/K) + log(U*(1-U)) + log(K) - log(mu_K)
                
                acc_DM_prob = min(1, exp(log_a_DM))
                acc_MHM = rand() < acc_DM_prob
            
                if acc_MHM
                    K_all[iter] = K_star
                    aK_all[iter] = 1
                    if K_star == 0
                        zetas_all[1:K_star,iter] = zeros(K_star)
                    else
                        zetas_all[1:K_star,iter] = zetas_star_rm[2:(K_star+1)]
                    end
                    xis_all[1:(K_star+2),iter] = xis_star_rm
                    K = K_star
                    zetas = zetas_star_rm[2:(K_star+1)]
                    xis = xis_star_rm
                    sigma_xi = sigma_xi_star
                else
                    K_all[iter] = K
                    aK_all[iter] = 0
                    zetas_all[1:K,iter] = zetas_star[2:(K+1)]
                    xis_all[1:(K+2),iter] = xis_star
                    zetas = zetas_star[2:(K+1)]
                    xis = xis_star
                    sigma_xi = sigma_xi_star
                end
            end
        else
            K_all[iter] = K
            aK_all[iter] = 0
            zetas_all[1:K,iter] = zetas_star[2:(K+1)]
            xis_all[1:(K+2),iter] = xis_star
            zetas = zetas_star[2:(K+1)]
            xis = xis_star
            sigma_xi = sigma_xi_star
        end
        
        # ----------------------------- Update alpha_tau (Eq. 6 in supplement) -----------------------------
        taus_ = [0.0; taus; tau]
        if H >= 0
            S_tau = sum(log.(diff(taus_)))
            
            # Propose on log-scale
            log_alpha_tau = log(alpha_tau)
            log_alpha_tau_star = rand(Normal(log_alpha_tau, c_alpha))
            alpha_tau_star = exp(log_alpha_tau_star)
            
            # Log acceptance ratio (Eq. 7)
            log_r_alpha = (a_tau - 1) * (log_alpha_tau_star - log_alpha_tau) - 
                          b_tau * (alpha_tau_star - alpha_tau) +
                          loggamma((H+1) * alpha_tau_star) - loggamma((H+1) * alpha_tau) -
                          (H+1) * (loggamma(alpha_tau_star) - loggamma(alpha_tau)) +
                          (alpha_tau_star - alpha_tau) * S_tau +
                          (log_alpha_tau_star - log_alpha_tau)  # Jacobian
            
            if rand() < min(1, exp(log_r_alpha))
                alpha_tau = alpha_tau_star
            end
        end
        alpha_tau_all[iter] = alpha_tau
        
        # ----------------------------- Update alpha_zeta -----------------------------
        zetas_ = [a; zetas; b]
        if K >= 0
            S_zeta = sum(log.(diff(zetas_)))
            
            log_alpha_zeta = log(alpha_zeta)
            log_alpha_zeta_star = rand(Normal(log_alpha_zeta, c_alpha))
            alpha_zeta_star = exp(log_alpha_zeta_star)
            
            log_r_alpha = (a_zeta - 1) * (log_alpha_zeta_star - log_alpha_zeta) - 
                          b_zeta * (alpha_zeta_star - alpha_zeta) +
                          loggamma((K+1) * alpha_zeta_star) - loggamma((K+1) * alpha_zeta) -
                          (K+1) * (loggamma(alpha_zeta_star) - loggamma(alpha_zeta)) +
                          (alpha_zeta_star - alpha_zeta) * S_zeta +
                          (log_alpha_zeta_star - log_alpha_zeta)
            
            if rand() < min(1, exp(log_r_alpha))
                alpha_zeta = alpha_zeta_star
            end
        end
        alpha_zeta_all[iter] = alpha_zeta
            
        # ----------------------------- Adaptive tuning -----------------------------
        if iter % n_report == 0
            if Adapt_C
                avg_gammas_aratio = mean(sum(agammas_all[:,(n_report*c_cnt+1):(n_report*(c_cnt+1))], dims=2) ./ n_report)
                avg_xis_aratio = mean(sum(axis_all[:,(n_report*c_cnt+1):(n_report*(c_cnt+1))],dims=2) ./ n_report)
                avg_betas_aratio = mean(sum(abetas_all[:,(n_report*c_cnt+1):(n_report*(c_cnt+1))],dims=2) ./ n_report) 
                
                if avg_gammas_aratio > AHigh
                    c_gamma = min(c_gamma*2, 2)
                end
                if avg_gammas_aratio < ALow
                    c_gamma = max(c_gamma/2, 0.0625)
                end
                if avg_xis_aratio > AHigh
                    c_xi = min(c_xi*2, 2)
                end
                if avg_xis_aratio < ALow
                    c_xi = max(c_xi/2, 0.0625)
                end
                if avg_betas_aratio > AHigh
                    c_beta = min(c_beta*2, 2)
                end
                if avg_betas_aratio < ALow
                    c_beta = max(c_beta/2, 0.0625)
                end
                c_cnt += 1
            else
                print("Iteration ", iter, " \n")
            end 
        end
        if show_progress
            next!(prog)
        end
    end

    results = Dict(
        "H"=>H_all,
        "taus"=>taus_all,
        "gammas"=>gammas_all,
        "sigmas_gamma"=>sigmas_gamma_all,
        "K"=>K_all,
        "zetas"=>zetas_all,
        "xis"=>xis_all,
        "sigmas_xi"=>sigmas_xi_all,
        "betas"=>betas_all,
        "tau"=> tau,
        "Tmax" => tau,
        "a"=> a,      
        "b"=> b,
        "ns" => NS,
        "burn_in" => BI,
        "alpha_tau" => alpha_tau_all,
        "alpha_zeta" => alpha_zeta_all
    )

    return results
end

##################################################################################################################
##################################################################################################################

mutable struct CoxPHWorkspace
    eta::Vector{Float64}
    knots_h::Vector{Float64}
    cumhaz_h::Vector{Float64}
    loglambda_values::Vector{Float64}  # logλ(Y) values (pre-allocated)
    Lambda_values::Vector{Float64}     # Λ(Y) values (pre-allocated)
end

function CoxPHWorkspace(n::Integer, Hmax::Integer)
    CoxPHWorkspace(
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, Hmax + 2),
        Vector{Float64}(undef, Hmax + 2),
        Vector{Float64}(undef, n),
        Vector{Float64}(undef, n),
    )
end

# logarithmic likelihood for CoxPH model (no nonlinear term here)
function coxph_loglkh_cal(X, Delta, Y,
                            betas,
                          tau, taus, gammas)
    ws = CoxPHWorkspace(length(Y), length(taus) + 2)
    return coxph_loglkh_cal(ws, X, Delta, Y, betas, tau, taus, gammas)
end

function coxph_loglkh_cal(ws::CoxPHWorkspace,
                          X, Delta, Y,
                          betas,
                          tau, taus, gammas)
    # Optimized version using in-place operations and pre-allocated buffers
    mul!(ws.eta, X, betas)

    # Compute loglambda_values and Lambda_values using in-place optimized functions
    loglambda_fun_est!(ws.loglambda_values, Y, tau, taus, gammas, ws.knots_h)
    Lambda_fun_est!(ws.Lambda_values, Y, tau, taus, gammas, ws.knots_h, ws.cumhaz_h)
    
    # Optimized vectorized log-likelihood calculation - avoid intermediate allocations
    loglkh = 0.0
    @inbounds @simd for i in eachindex(Delta)
        exp_val = exp(ws.eta[i])
        loglkh += Delta[i] * (ws.loglambda_values[i] + ws.eta[i]) - 
                  ws.Lambda_values[i] * exp_val
    end
    return loglkh
end

# Reversible-jump sampler for baseline hazard with linear covariates (CoxPH variant).
function RJMCMC_CoxPH(
        X, Z, Delta, Y,
        random_seed;
        ns::Int=DEFAULT_NS,
        burn_in::Int=DEFAULT_BURN_IN,
        RJMCMC_indicator::Bool=true,
        Adapt_C::Bool=true,
        Hmax::Int=DEFAULT_HMAX,
        show_progress::Bool=true
     )

    NS = ns
    BI = burn_in

	X = hcat(X,Z)
    
    # ---------------------------- Hyper-Parameters ----------------------------    
    # Hmax
    Hmax = Hmax
    # dimension of beta
    dim_beta = size(X,2)
    ws = CoxPHWorkspace(length(Y), Hmax)
    
    # hyper-parameters
    # obtain the observed time
    obs_time = (Y .* Delta) 
    obs_time = obs_time[obs_time .> 0]
    tau = maximum(obs_time)+1e-5
    
    # prior parameter for number of probability
    mu_H = 1

    # initial H
    H = 0
    
    # coefficients
    c_gamma = 1    # for gamma
    c_xi = 1
    c_beta = 1     # for betas
    c_cnt = 0
    AHigh = 0.4
    ALow = 0.2
    
    # number of reports
    n_report = 250
    
    # birth probability
    r_HB = 0.5
    
    # ---------------------------- Coefficient Variables ----------------------------
    # PH Model coefficients
    betas_all = zeros(dim_beta,NS)
    
    # hazard function coefficients
    H_all = zeros(NS)
    taus_all = zeros(Hmax,NS)     # points
    gammas_all = zeros(Hmax+2,NS) # slope
    
    
    # variance 
    sigmas_gamma_all = zeros(NS)
    sigmas_xi_all = zeros(NS)
    
    # ---------------------------- Acceptance Variables ----------------------------
    # acceptance rate calculation
    
    # PH Model coefficients
    abetas_all = zeros(dim_beta,NS)
    
    # hazard function coefficients
    aH_all = zeros(NS)
    agammas_all = zeros(Hmax+2,NS)  # slope
    
    # ------------------ Prior Specification ------------------ 
    
    Random.seed!(random_seed)
    
    # Create candidate sets (matching NonLinear1 exactly)
    Hcan = 20  # Candidate set size for baseline hazard knots
    taus_can = LinRange(0, tau, Hcan+2)[2:(Hcan+1)]
    
    # prior: baseline hazard using candidate set method (matching NonLinear1)
    if H > 0
        taus = sort(sample(taus_can, H, replace=false))
    else
        taus = Float64[]  # Empty when H=0
    end
    taus_ = [0 taus' tau][1,:]
    
    gammas = zeros(H+2)
    gammas[1] = rand(Normal(0,5))
    for i in 2:(H+2)
        gammas[i] = rand(Normal(gammas[i-1],1))
    end
    
    # beta: coefficients
    betas = zeros(dim_beta)
    
    # ------------------ Update Coefficient ------------------ 
    H_all[1] = H
    taus_all[1:H,1] = taus
    gammas_all[1:(H+2),1] = gammas
    betas_all[:,1] = betas
    sigmas_gamma_all[1] = 1
    
    # ------------------ Update Acceptance Count ------------------ 
    aH_all[1] = 1
    #ataus_all[1:H,1] .= 1  
    agammas_all[1:(H+2),1] .= 1 
    abetas_all[:,1] .= 1

    Random.seed!(random_seed)

    # Only create progress bar if explicitly enabled
    if show_progress
        prog = Progress(NS - 1; desc="MCMC")
    end
    for iter in 2:NS

        # ----------------------------- Update Gamma ----------------------------- 
        gammas_star = copy(gammas)
        agamma_vec = zeros(size(gammas)[1])
        sigma_gamma = sigmas_gamma_all[iter-1]
        
        # Pre-compute components that don't change during gamma updates
        mul!(ws.eta, X, betas)
        
        for h in 1:(H+2)       
            # compute the denominator - use optimized logpdf_normal
            log_prob_de = logpdf_normal(gammas_star[1], 0.0, 5.0)
            @inbounds for hh in 2:(H+2)
                log_prob_de += logpdf_normal(gammas_star[hh], gammas_star[hh-1], sigma_gamma)
            end
            
            # Compute loglambda and Lambda for current gammas
            loglambda_fun_est!(ws.loglambda_values, Y, tau, taus, gammas_star, ws.knots_h)
            Lambda_fun_est!(ws.Lambda_values, Y, tau, taus, gammas_star, ws.knots_h, ws.cumhaz_h)
            log_de = log_prob_de + compute_coxph_loglkh_optimized(Delta, ws.loglambda_values, ws.eta, ws.Lambda_values)
            
            # compute the numerator
            # propose a new gamma
            gamma_h_new = rand(Uniform(gammas[h]-c_gamma, gammas[h]+c_gamma))
            gammas_star[h] = gamma_h_new
            
            log_prob_num = logpdf_normal(gammas_star[1], 0.0, 5.0)
            @inbounds for hh in 2:(H+2)
                log_prob_num += logpdf_normal(gammas_star[hh], gammas_star[hh-1], sigma_gamma)
            end
            
            # Recompute loglambda and Lambda for proposed gammas
            loglambda_fun_est!(ws.loglambda_values, Y, tau, taus, gammas_star, ws.knots_h)
            Lambda_fun_est!(ws.Lambda_values, Y, tau, taus, gammas_star, ws.knots_h, ws.cumhaz_h)
            log_num = log_prob_num + compute_coxph_loglkh_optimized(Delta, ws.loglambda_values, ws.eta, ws.Lambda_values)
            
            # acceptance ratio
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            # accept or not
            agamma_vec[h] = acc = rand() < acc_prob
            if !acc
                # Revert the change
                gammas_star[h] = gammas[h]
            end
        end
        
        # update the acceptance vector
        agammas_all[1:(H+2),iter] = agamma_vec
        
        # ----------------------------- Update Sigma_Gamma -----------------------------
        shape_param = 0.5 * H + 0.5
        # Optimized: compute sum of squared differences directly
        scale_param = 0.0
        @inbounds @simd for i in 1:(H+1)
            diff_val = gammas_star[i+1] - gammas_star[i]
            scale_param += diff_val * diff_val
        end
        scale_param *= 0.5
        sigma_gamma_star = sqrt(rand(InverseGamma(shape_param, scale_param)))
        
        # update the sequence
        sigmas_gamma_all[iter] = sigma_gamma_star
        
        # ----------------------------- Update Tau -----------------------------
        taus_ = [0 taus' tau][1,:]
        taus_star = copy(taus_)
        taus_star_replace = copy(taus_)

		if H > 0
             hc = rand(1:H)
             tau_hc_star = rand(Uniform(taus_star[hc],taus_star[hc+2]))
             taus_star_replace[hc+1] = tau_hc_star
             log_de = coxph_loglkh_cal(ws, X,Delta,Y,  
								betas,
								tau,
								taus_star[2:(H+1)], 
								gammas_star) +
                            log(taus_star[hc+2] - taus_star[hc+1]) + log(taus_star[hc+1] - taus_star[hc])
            
             log_num = coxph_loglkh_cal(ws, X,Delta,Y, 
								betas,
								tau,
								taus_star_replace[2:(H+1)], 
								gammas_star) + 
                            log(taus_star[hc+2] - tau_hc_star) + log(tau_hc_star - taus_star[hc])
                    
             aratio = exp(log_num - log_de)
             acc_prob = min(1, aratio)
             # accept or not
             acc = rand() < acc_prob
             # update the taus
             taus_star[hc+1] = acc * tau_hc_star + (1-acc) * taus_[hc+1] 
        end
        
        
        # # ------------ RJMCMC: Perform RJ-Update for (H, taus, gammas)------------
    
        # check the performance
       if H == 0
            r_HB_star = 1
            H_star = H + 1
        elseif H == Hmax
            r_HB_star = 0
            H_star = H - 1
        else
            r_HB_star = r_HB
            H_star = H + 2*Int(rand(Bernoulli(0.5))) - 1
        end
        
        if RJMCMC_indicator
            if H_star > H
                # ------------ codes for H_star > H (generate a new knot) ------------
            
                # sample tau from the unselected observed time
                tau_star = rand(Uniform(0,tau))
                
                # merge new tau into a new list 
                taus_star_add = sort([taus_star; tau_star])
                
                # propose a new gamma
                h = sum(taus_star .< tau_star) 
                Ah = (gammas_star[h+1] - gammas_star[h]) / (taus_star[h+1] - taus_star[h])
                U = rand()
                gamma_star = gammas_star[h] + (tau_star - taus_star[h]) * (Ah - (taus_star[h+1] - tau_star) 
                    / (taus_star[h+1] - taus_star[h]) * log((1-U)/U))
                # add the new gamma into the list
                gammas_star_add = [gammas_star[1:h]; gamma_star; gammas_star[(h+1):(H_star+1)]]
                
                # acceptance rate
                log_a_BM = coxph_loglkh_cal(ws, X,Delta,Y, 
                                            betas,
                                            tau,
                                            taus_star_add[2:(H_star+1)], 
                                            gammas_star_add) - 
                           coxph_loglkh_cal(ws, X,Delta,Y, 
                                        betas,
                                        tau,
                                        taus_star[2:(H+1)], 
                                        gammas_star) + 
                       log(2*H+3) + log(2*H+2) + log(tau_star-taus_star[h]) + log(taus_star[h+1]-tau_star) + 
                       logpdf_normal(gamma_star, gammas_star[h], sigma_gamma_star) +
                       logpdf_normal(gammas_star[h+1], gamma_star, sigma_gamma_star) - 
                       2*log(tau) - log(taus_star[h+1] - taus_star[h]) - 
                       logpdf_normal(gammas_star[h+1], gammas_star[h], sigma_gamma_star) +
                       log((1-r_HB)/(H+1)) - log(r_HB_star/tau) - log(U*(1-U)) + log(mu_H) - log(H+1)
    
                acc_BM_prob = min(1, exp(log_a_BM))
                
                # accept or not for the Metropolis-Hastings' Move
                acc_MHM = rand() < acc_BM_prob
                # acc_MHM = false
                
                # update the coefficients
                if acc_MHM
                    H_all[iter] = H_star
                    aH_all[iter] = 1
                    taus_all[1:H_star,iter] = taus_star_add[2:(H_star+1)]
                    gammas_all[1:(H_star+2),iter] = gammas_star_add
            
                    H = H_star
                    taus = taus_star_add[2:(H_star+1)]
                    gammas = gammas_star_add
                    sigma_gamma = sigma_gamma_star
                else
                    H_all[iter] = H
                    aH_all[iter] = 0
                    taus_all[1:H,iter] = taus_star[2:(H+1)]
                    gammas_all[1:(H+2),iter] = gammas_star
            
                    taus = taus_star[2:(H+1)]
                    gammas = gammas_star
                    sigma_gamma = sigma_gamma_star
                end
                    
            else 
                # ------------ codes for H_star < H (remove a new knot) ------------
            
                # select a random value
                h = rand([1:1:H;])
                
                # remove an element from taus
                taus_star_rm = copy(taus_star)
                taus_star_rm = deleteat!(taus_star_rm, h+1)
                
                # remove an element from gammas
                gammas_star_rm = copy(gammas_star)
                gammas_star_rm = deleteat!(gammas_star_rm, h+1)

                A_hp1 = (gammas_star[h+2]-gammas_star[h+1])/(taus_star[h+2]-taus_star[h+1])
				A_h = (gammas_star[h+1]-gammas_star[h])/(taus_star[h+1]-taus_star[h])
				U = exp(A_h)/(exp(A_h)+exp(A_hp1))
                
                # logarithmic 
                log_a_DM = coxph_loglkh_cal(ws, X,Delta,Y, 
                                    betas,
                                    tau,
                                    taus_star_rm[2:(H_star+1)], 
                                    gammas_star_rm) - 
                        coxph_loglkh_cal(ws, X,Delta,Y, 
                                    betas,
                                    tau,
                                    taus_star[2:(H+1)], 
                                    gammas_star) + 
                       2*log(tau) + log(taus_star_rm[h+1]-taus_star_rm[h]) + 
                       logpdf_normal(gammas_star_rm[h+1], gammas_star_rm[h], sigma_gamma_star) - 
                       log(2*H+1) - log(2*H) - 
                       log(taus_star[h+2]-taus_star[h+1]) - log(taus_star[h+1]-taus_star[h]) - 
                       logpdf_normal(gammas_star[h+2], gammas_star[h+1], sigma_gamma_star) -
                       logpdf_normal(gammas_star[h+1], gammas_star[h], sigma_gamma_star) +
                       log(r_HB/tau) - log((1-r_HB_star)/H) + log(U*(1-U)) + log(H) - log(mu_H)
                
                acc_DM_prob = min(1, exp(log_a_DM))
                
                # accept or not for the Metropolis-Hastings' Move
                acc_MHM = rand() < acc_DM_prob
            
                # update the coefficients
                if acc_MHM
                    H_all[iter] = H_star
                    aH_all[iter] = 1
                    taus_all[1:H_star,iter] = taus_star_rm[2:(H_star+1)]
                    gammas_all[1:(H_star+2),iter] = gammas_star_rm
            
                    H = H_star
                    taus = taus_star_rm[2:(H_star+1)]
                    gammas = gammas_star_rm
                    sigma_gamma = sigma_gamma_star
                else
                    H_all[iter] = H
                    aH_all[iter] = 0
                    taus_all[1:H,iter] = taus_star[2:(H+1)]
                    gammas_all[1:(H+2),iter] = gammas_star
            
                    taus = taus_star[2:(H+1)]
                    gammas = gammas_star
                    sigma_gamma = sigma_gamma_star
                end
            end
        else
            H_all[iter] = H
            aH_all[iter] = 0
            taus_all[1:H,iter] = taus_star[2:(H+1)]
            gammas_all[1:(H+2),iter] = gammas_star
    
            taus = taus_star[2:(H+1)]
            gammas = gammas_star
            sigma_gamma = sigma_gamma_star
        end
    
        # ----------------------------- Update Beta -----------------------------
        betas_star = copy(betas)
        abeta_vec = zeros(size(betas)[1])
        
        # Pre-compute baseline components that don't change during beta updates
        mul!(ws.eta, X, betas_star)
        loglambda_fun_est!(ws.loglambda_values, Y, tau, taus, gammas, ws.knots_h)
        Lambda_fun_est!(ws.Lambda_values, Y, tau, taus, gammas, ws.knots_h, ws.cumhaz_h)
        
        for j in 1:dim_beta
            # compute the denominator (current state)
            log_de = compute_coxph_loglkh_optimized(Delta, ws.loglambda_values, ws.eta, ws.Lambda_values)
            
            # propose a new beta - use betas[j] (original state) to match NonLinear1 exactly
            beta_j_new = rand(Uniform(betas[j]-c_beta, betas[j]+c_beta))
            
            # Incremental update: only change Xbeta for beta_j
            beta_diff = beta_j_new - betas_star[j]
            @inbounds @simd for i in eachindex(ws.eta)
                ws.eta[i] += X[i, j] * beta_diff
            end
    
            # compute the numerator (proposed state)
            log_num = compute_coxph_loglkh_optimized(Delta, ws.loglambda_values, ws.eta, ws.Lambda_values)
            
            # acceptance ratio
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            # accept or not
            abeta_vec[j] = acc = rand() < acc_prob
            if !acc
                # Revert the change
                @inbounds @simd for i in eachindex(ws.eta)
                    ws.eta[i] -= X[i, j] * beta_diff
                end
                betas_star[j] = betas[j]
            else
                betas_star[j] = beta_j_new
            end
        end
        
        # update the coefficients
        betas_all[:,iter] = betas_star
        abetas_all[:,iter] = abeta_vec
        betas = betas_star
    
        if iter % n_report == 0
    
            if Adapt_C
                avg_gammas_aratio = mean(sum(agammas_all[:,(n_report*c_cnt+1):(n_report*(c_cnt+1))], dims=2) ./ n_report)
                avg_betas_aratio = mean(sum(abetas_all[:,(n_report*c_cnt+1):(n_report*(c_cnt+1))],dims=2) ./ n_report) 
                
                if avg_gammas_aratio > AHigh
                    c_gamma = min(c_gamma*2, 2)
                end
                
                if avg_gammas_aratio < ALow
                    c_gamma = max(c_gamma/2, 0.0625)
                end
                
                if avg_betas_aratio > AHigh
                    c_beta = min(c_beta*2, 2)
                end
                
                if avg_betas_aratio < ALow
                    c_beta = max(c_beta/2, 0.0625)
                end
                
                c_cnt += 1
                
            else
                print("Iteration ", iter, " \n")
            end 
        end
        if show_progress
            next!(prog)
        end
    end

    results = Dict(
        "H"=>H_all,
        "taus"=>taus_all,
        "gammas"=>gammas_all,
        "sigmas_gamma"=>sigmas_gamma_all,
        "betas"=>betas_all,
        "tau"=> tau,
        "ns" => NS,
        "burn_in" => BI
    )

    return results
end

# Survival prediction for nonlinear model: returns posterior mean and 95% bands.
function St_pred(t,
                 X_, Z_,
                 results)

    betas_all, K_all, zetas_all, xis_all, H_all, taus_all, gammas_all = results["betas"], results["K"], results["zetas"], results["xis"],
                                                                         results["H"], results["taus"], results["gammas"]
    a, b, tau = results["a"], results["b"], results["tau"]

    ns = get(results, "ns", size(betas_all, 2))
    bi = get(results, "burn_in", ns ÷ 2)
    n_samples = ns - bi
    n_t = length(t)

    # Pre-allocate buffers for reuse
    knots_g_buf = Vector{Float64}(undef, DEFAULT_KMAX + 2)
    knots_h_buf = Vector{Float64}(undef, DEFAULT_HMAX + 2)
    cumhaz_buf = Vector{Float64}(undef, DEFAULT_HMAX + 2)
    g_buf = Vector{Float64}(undef, 1)  # Single value for scalar Z_
    Lambda_buf = Vector{Float64}(undef, n_t)

    St_all = zeros(n_t, n_samples)
    # Pre-allocate Z buffer to avoid creating new array each iteration
    Z_buf = [Z_]
    
    col_idx = 0
    for i in (bi+1):ns
        col_idx += 1
        Xbeta = dot(X_, @view(betas_all[:, i]))
        
        K = Int(K_all[i])
        # Reuse Z_buf instead of creating [Z_] each time
        g_fun_est!(g_buf, Z_buf, a, b, @view(zetas_all[1:K, i]), @view(xis_all[1:(K+2), i]), knots_g_buf)
        g = g_buf[1]
        H = Int(H_all[i])
        Lambda_fun_est!(Lambda_buf, t, tau, @view(taus_all[1:H, i]), @view(gammas_all[1:(H+2), i]), knots_h_buf, cumhaz_buf)
        @inbounds for j in 1:n_t
            St_all[j, col_idx] = exp(-exp(Xbeta + g) * Lambda_buf[j])
        end
    end

    St_avg = mean(St_all, dims=2)
    St_lb = vquantile!(St_all, 0.025, dims=2)
    St_ub = vquantile!(St_all, 0.975, dims=2)
    
    return St_avg, St_lb, St_ub
end

mutable struct StPredMeanWorkspace
    knots_g_buf::Vector{Float64}
    knots_h_buf::Vector{Float64}
    cumhaz_buf::Vector{Float64}
    St_sum::Vector{Float64}
end

function StPredMeanWorkspace(n_t::Integer; Hmax::Integer=DEFAULT_HMAX, Kmax::Integer=DEFAULT_KMAX)
    StPredMeanWorkspace(
        Vector{Float64}(undef, Kmax + 2),
        Vector{Float64}(undef, Hmax + 2),
        Vector{Float64}(undef, Hmax + 2),
        zeros(Float64, n_t),
    )
end

# Optimized version of St_pred that only computes mean (for IBS)
# This avoids storing the full St_all matrix and computing quantiles
# Supports sparse sampling for faster IBS computation when n_samples is large
function St_pred_mean_only!(ws::StPredMeanWorkspace,
                            t,
                            X_, Z_,
                            results;
                            thin::Int=1)
    betas_all, K_all, zetas_all, xis_all, H_all, taus_all, gammas_all = results["betas"], results["K"], results["zetas"], results["xis"],
                                                                         results["H"], results["taus"], results["gammas"]
    a, b, tau = results["a"], results["b"], results["tau"]

    ns = get(results, "ns", size(betas_all, 2))
    bi = get(results, "burn_in", ns ÷ 2)
    n_t = length(t)
    if length(ws.St_sum) != n_t
        ws.St_sum = zeros(Float64, n_t)
    else
        fill!(ws.St_sum, 0.0)
    end
    
    # Sparse sampling: only use every 'thin'-th sample for faster computation
    # This reduces computation by thin× while maintaining accuracy
    count = 0
    for i in (bi+1):ns
        if (i - bi - 1) % thin != 0
            continue
        end
        count += 1
        
        Xbeta = dot(X_, @view(betas_all[:, i]))
        
        K = Int(K_all[i])
        len_g = fill_knots!(ws.knots_g_buf, Float64(a), @view(zetas_all[1:K, i]), Float64(b))
        g = piecewise_linear_at(Float64(Z_), ws.knots_g_buf, @view(xis_all[1:(K+2), i]), len_g)
        H = Int(H_all[i])
        len_h = fill_knots!(ws.knots_h_buf, 0.0, @view(taus_all[1:H, i]), Float64(tau))
        gammas_view = @view(gammas_all[1:(H+2), i])
        prepare_cumhaz!(ws.cumhaz_buf, ws.knots_h_buf, gammas_view, len_h)
        
        # Batch compute Lambda for all time points - optimized with @simd
        exp_Xbeta_plus_g = exp(Xbeta + g)
        @inbounds @simd for j in 1:n_t
            tj = Float64(t[j])
            if tj > tau
                tj = Float64(tau)
            end
            Λ = cumhaz_at(tj, ws.knots_h_buf, ws.cumhaz_buf, gammas_view, len_h)
            ws.St_sum[j] += exp(-exp_Xbeta_plus_g * Λ)
        end
    end
    
    inv_n = 1.0 / count
    @inbounds @simd for j in 1:n_t
        ws.St_sum[j] *= inv_n
    end
    return ws.St_sum
end

function St_pred_mean_only(t,
                           X_, Z_,
                           results)
    ws = StPredMeanWorkspace(length(t))
    return St_pred_mean_only!(ws, t, X_, Z_, results)
end

# Survival prediction for CoxPH variant (linear Z term).
function coxph_St_pred(t,
                 X_, Z_,
                 results)

    betas_all, H_all, taus_all, gammas_all = results["betas"], results["H"], results["taus"], results["gammas"]
    tau = results["tau"]

    ns = get(results, "ns", size(betas_all, 2))
    bi = get(results, "burn_in", ns ÷ 2)
    n_samples = ns - bi
    n_t = length(t)

	d = size(X_)[1]
    
    # Pre-allocate buffers for reuse
    knots_h_buf = Vector{Float64}(undef, DEFAULT_HMAX + 2)
    cumhaz_buf = Vector{Float64}(undef, DEFAULT_HMAX + 2)
    Lambda_buf = Vector{Float64}(undef, n_t)

    St_all = zeros(n_t, n_samples)
    col_idx = 0
    for i in (bi+1):ns
        col_idx += 1
        Xbeta = sum(X_.*betas_all[1:d,i])
        beta_z = betas_all[d+1,i]
        H = Int(H_all[i])
        Lambda_fun_est!(Lambda_buf, t, tau, taus_all[1:H,i], gammas_all[1:(H+2),i], knots_h_buf, cumhaz_buf)
        @inbounds for j in 1:n_t
            St_all[j, col_idx] = exp(-exp(Xbeta + Z_ * beta_z) * Lambda_buf[j])
        end
    end

    St_avg = mean(St_all, dims=2)
    St_lb = vquantile!(St_all, 0.025, dims=2)
    St_ub = vquantile!(St_all, 0.975, dims=2)
    
    return St_avg, St_lb, St_ub
end

mutable struct CoxPredMeanWorkspace
    knots_h_buf::Vector{Float64}
    cumhaz_buf::Vector{Float64}
    St_sum::Vector{Float64}
end

function CoxPredMeanWorkspace(n_t::Integer; Hmax::Integer=DEFAULT_HMAX)
    CoxPredMeanWorkspace(
        Vector{Float64}(undef, Hmax + 2),
        Vector{Float64}(undef, Hmax + 2),
        zeros(Float64, n_t),
    )
end

function coxph_St_pred_mean_only!(ws::CoxPredMeanWorkspace,
                                  t,
                                  X_, Z_,
                                  results;
                                  thin::Int=1)
    betas_all, H_all, taus_all, gammas_all = results["betas"], results["H"], results["taus"], results["gammas"]
    tau = results["tau"]

    ns = get(results, "ns", size(betas_all, 2))
    bi = get(results, "burn_in", ns ÷ 2)
    n_t = length(t)

    if length(ws.St_sum) != n_t
        ws.St_sum = zeros(Float64, n_t)
    else
        fill!(ws.St_sum, 0.0)
    end

    d = length(X_)
    # Note: In CoxPH, X and Z are combined into a single covariate vector
    # betas_all has dim_beta columns, where dim_beta = size(X, 2) + 1 (last one is Z coefficient)

    # Sparse sampling for faster computation
    count = 0
    for i in (bi+1):ns
        if (i - bi - 1) % thin != 0
            continue
        end
        count += 1
        
        # Compute Xbeta + Z_ * beta_z (matching coxph_St_pred logic)
        Xbeta = dot(X_, @view(betas_all[1:d, i]))
        beta_z = betas_all[d+1, i]
        H = Int(H_all[i])
        len_h = fill_knots!(ws.knots_h_buf, 0.0, @view(taus_all[1:H, i]), Float64(tau))
        gammas_view = @view(gammas_all[1:(H+2), i])
        prepare_cumhaz!(ws.cumhaz_buf, ws.knots_h_buf, gammas_view, len_h)
        # Pre-compute exp(Xbeta + Z_ * beta_z) once per sample
        exp_Xbeta_plus_Z = exp(Xbeta + Z_ * beta_z)
        @inbounds @simd for j in 1:n_t
            tj = Float64(t[j])
            if tj > tau
                tj = Float64(tau)
            end
            Λ = cumhaz_at(tj, ws.knots_h_buf, ws.cumhaz_buf, gammas_view, len_h)
            ws.St_sum[j] += exp(-exp_Xbeta_plus_Z * Λ)
        end
    end

    inv_n = 1.0 / count
    @inbounds @simd for j in 1:n_t
        ws.St_sum[j] *= inv_n
    end
    return ws.St_sum
end

function coxph_St_pred_mean_only(t, X_, Z_, results)
    ws = CoxPredMeanWorkspace(length(t))
    return coxph_St_pred_mean_only!(ws, t, X_, Z_, results)
end

# Kaplan–Meier estimator at time t (used for IBS weighting).
function KM_est(t,Y,Delta)
    sort_idx = sortperm(Y)
    Y = Y[sort_idx]
    Delta = Delta[sort_idx]
    S_KM = 1
    i = sum(Y.<=t)
    for ii in 1:i
        if Int(Delta[ii]) == 1
            di = sum(Y.==Y[ii])
            ni = sum(Y.>=Y[ii])
            S_KM *= (1-di/ni) 
        end
    end
    return S_KM
end

# Optimized batch KM estimator: pre-compute KM for all unique times
# Uses sorted array and binary search for efficient lookup
function KM_est_batch(times::Vector{Float64}, Y::Vector{Float64}, Delta)
    # Sort data
    sort_idx = sortperm(Y)
    Y_sorted = Y[sort_idx]
    Delta_sorted = Delta[sort_idx]
    
    # Get unique event times (sorted)
    unique_times = sort(unique(Y_sorted[Delta_sorted.==1]))
    n_unique = length(unique_times)
    
    # Pre-compute KM at unique times
    km_values_at_events = Vector{Float64}(undef, n_unique)
    S_KM = 1.0
    
    event_idx = 1
    for (ii, y_val) in enumerate(Y_sorted)
        if Int(Delta_sorted[ii]) == 1
            di = sum(Y_sorted.==y_val)
            ni = sum(Y_sorted.>=y_val)
            S_KM *= (1 - di/ni)
            if event_idx <= n_unique && unique_times[event_idx] == y_val
                km_values_at_events[event_idx] = S_KM
                event_idx += 1
            end
        end
    end
    
    # Interpolate KM for requested times using binary search
    km_values = Vector{Float64}(undef, length(times))
    max_time = maximum(Y_sorted)
    max_km = n_unique > 0 ? km_values_at_events[n_unique] : 1.0
    
    @inbounds for (i, t) in enumerate(times)
        if t <= 0
            km_values[i] = 1.0
        elseif t >= max_time
            km_values[i] = max_km
        else
            # Binary search for the largest event time <= t
            idx = searchsortedlast(unique_times, t)
            if idx == 0
                km_values[i] = 1.0
            else
                km_values[i] = km_values_at_events[idx]
            end
        end
    end
    
    return km_values
end

# Integrated Brier Score with inverse-probability-of-censoring weighting.
# Optimized version: matches model0109.jl logic but uses batch KM computation for speed
function IBS(Y_train, Delta_train, 
              X_train, Z_train, 
              Y_test, Delta_test,
              X_test, Z_test,
              results, model,
              n_int=200, 
              random_seed=2024;
              show_progress=false)

    n_dag = size(Y_test)[1]  #size of the validating dataset
    tau = results["tau"]
    
    # Pre-compute KM estimates for all possible times (for batch efficiency)
    # Collect all possible times: Y_test values and potential t_int values
    # We'll use a grid approach to pre-compute KM for efficiency
    all_possible_times = unique(vcat(Y_test[Y_test .<= tau], collect(range(0, tau, length=min(1000, n_int*5)))))
    sort!(all_possible_times)
    G_all = KM_est_batch(all_possible_times, Y_test, 1 .- Delta_test)
    max_time = maximum(all_possible_times)
    max_G = G_all[end]
    
    # Helper function to get KM value efficiently using binary search
    function get_km(t)
        if t <= 0
            return 1.0
        elseif t >= max_time
            return max_G
        else
            # Binary search for the largest time <= t
            idx = searchsortedlast(all_possible_times, t)
            return idx == 0 ? 1.0 : G_all[idx]
        end
    end
 
    # Aggressive importance sampling strategy for IBS calculation
    # Inspired by model1229.jl: use event times + adaptive grid for maximum speedup
    ns = get(results, "ns", size(results["betas"], 2))
    bi = get(results, "burn_in", ns ÷ 2)
    n_samples = ns - bi
    
    # Optimized thinning strategy for NS=10000 (n_samples=5000)
    # More aggressive thinning for large n_samples to maintain speed
    if n_samples >= 4000
        # For large n_samples (NS=10000), use more aggressive thinning
        if n_int <= 30
            thin_factor = 5  # Increased from 2
        elseif n_int <= 50
            thin_factor = 10  # Increased from 4 for NS=10000
        elseif n_int <= 100
            thin_factor = 15  # Increased from 6
        elseif n_int <= 200
            thin_factor = 25  # Increased from 10
        else
            thin_factor = max(25, div(n_int, 20))  # More aggressive scaling
        end
    else
        # Original strategy for smaller n_samples
        if n_int <= 30
            thin_factor = 2
        elseif n_int <= 50
            thin_factor = 4
        elseif n_int <= 100
            thin_factor = 6
        elseif n_int <= 200
            thin_factor = 10
        else
            thin_factor = max(10, div(n_int, 25))
        end
    end
    
    # Safety: ensure we use at least 100 samples for stability
    # For NS=10000, we can use fewer samples due to better mixing
    min_samples = n_samples >= 4000 ? 80 : 100
    max_thin = max(1, div(n_samples, min_samples))
    thin_factor = min(thin_factor, max_thin)
    
    rng = MersenneTwister(random_seed)
    integrals = zeros(Float64, n_dag)
    
    if show_progress
        prog = Progress(n_dag; desc="IBS")
    end

    # Pre-extract event times from test set (used for importance sampling)
    event_times_all = Y_test[(Delta_test .== 1) .& (Y_test .<= tau)]
    event_times_unique = sort(unique(event_times_all))
    
    for i_dag in 1:n_dag
        Y_ = Float64(Y_test[i_dag])
        Delta_ = Delta_test[i_dag]
        X_ = @view X_test[i_dag, :]
        Z_ = Float64(Z_test[i_dag])
        
        # Aggressive importance sampling: prioritize event times for ALL n_int values
        # Strategy inspired by model1229.jl: use actual event times as primary sampling points
        # This dramatically reduces computation while maintaining accuracy
        
        # Always use event times when available, regardless of n_int
        t_event = filter(t -> t <= tau, event_times_unique)
        n_event = length(t_event)
        
        if n_event > 0
            # We have event times: use them as primary sampling points
            if n_event >= n_int
                # More events than needed: sample evenly
                idx = round.(Int, range(1, n_event, length=n_int))
                t_int = t_event[idx]
            else
                # Need additional points: combine events with adaptive grid
                n_grid = n_int - n_event
                
                # Adaptive grid: denser in early times where events are more frequent
                # For small n_int (like 50), use simpler strategy
                if n_int <= 50
                    # Simple strategy: 60% early, 40% late
                    early_frac = 0.6
                    n_early = div(n_grid * 3, 5)  # 60% early
                    n_late = n_grid - n_early
                    
                    if n_early > 0
                        early_grid = collect(range(0.0, tau * early_frac, length=n_early + 1))[1:end-1]
                    else
                        early_grid = Float64[]
                    end
                    if n_late > 0
                        late_grid = collect(range(tau * early_frac, tau, length=n_late + 1))[1:end-1]
                    else
                        late_grid = Float64[]
                    end
                    
                    # Combine events + grid, sort and deduplicate
                    t_int_all = vcat(t_event, early_grid, late_grid)
                    sort!(t_int_all)
                    # Remove points too close together
                    t_int = Float64[]
                    push!(t_int, t_int_all[1])
                    for i in 2:length(t_int_all)
                        if t_int_all[i] - t_int[end] > 1e-6
                            push!(t_int, t_int_all[i])
                        end
                    end
                    # Trim to n_int if needed
                    if length(t_int) > n_int
                        # Keep all events, sample grid points
                        t_grid = setdiff(t_int, t_event)
                        sort!(t_grid)
                        n_grid_keep = n_int - n_event
                        if n_grid_keep > 0 && length(t_grid) >= n_grid_keep
                            idx_grid = round.(Int, range(1, length(t_grid), length=n_grid_keep))
                            t_int = vcat(t_event, t_grid[idx_grid])
                        else
                            t_int = vcat(t_event, t_grid)[1:min(n_int, length(vcat(t_event, t_grid)))]
                        end
                        sort!(t_int)
                    end
                else
                    # For larger n_int: use more sophisticated strategy
                    early_idx = max(1, div(n_event * 7, 10))
                    early_cutoff = t_event[early_idx]
                    n_early_grid = div(n_grid * 2, 3)  # 2/3 of grid points in early region
                    n_late_grid = n_grid - n_early_grid
                    
                    # Early region grid
                    if n_early_grid > 0
                        early_grid = collect(range(0.0, early_cutoff, length=n_early_grid + 2))[2:(end-1)]
                    else
                        early_grid = Float64[]
                    end
                    # Late region grid
                    if n_late_grid > 0
                        late_grid = collect(range(early_cutoff, tau, length=n_late_grid + 2))[2:(end-1)]
                    else
                        late_grid = Float64[]
                    end
                    
                    # Combine: events + grid points, then sort and deduplicate
                    t_int_all = vcat(t_event, early_grid, late_grid)
                    sort!(t_int_all)
                    # Remove points too close together (within 1e-6)
                    t_int = Float64[]
                    if length(t_int_all) > 0
                        push!(t_int, t_int_all[1])
                        for i in 2:length(t_int_all)
                            if t_int_all[i] - t_int[end] > 1e-6
                                push!(t_int, t_int_all[i])
                            end
                        end
                    end
                    # Trim to n_int if needed (keep all events)
                    if length(t_int) > n_int
                        # Keep all events, sample grid points
                        t_grid = setdiff(t_int, t_event)
                        sort!(t_grid)
                        n_grid_keep = n_int - n_event
                        if n_grid_keep > 0 && length(t_grid) >= n_grid_keep
                            idx_grid = round.(Int, range(1, length(t_grid), length=n_grid_keep))
                            t_int = vcat(t_event, t_grid[idx_grid])
                        else
                            t_int = vcat(t_event, t_grid)[1:min(n_int, length(vcat(t_event, t_grid)))]
                        end
                        sort!(t_int)
                    end
                end
            end
        else
            # No events: use adaptive grid
            if n_int <= 50
                # Simple adaptive grid for small n_int
                early_frac = 0.6
                n_early = div(n_int * 3, 5)
                n_late = n_int - n_early
                if n_early > 0
                    early_grid = collect(range(0.0, tau * early_frac, length=n_early + 1))[1:end-1]
                else
                    early_grid = Float64[]
                end
                if n_late > 0
                    late_grid = collect(range(tau * early_frac, tau, length=n_late + 1))[1:end-1]
                else
                    late_grid = Float64[]
                end
                t_int = vcat(early_grid, late_grid)
            else
                # Use random sampling for larger n_int when no events
                t_int = rand(rng, Uniform(0, tau), n_int)
                sort!(t_int)
            end
        end
        
        n_t_int = length(t_int)
        
        # Allocate workspace for actual size
        nonlinear_ws = model == "nonlinear" ? StPredMeanWorkspace(n_t_int) : nothing
        coxph_ws = model == "coxph" ? CoxPredMeanWorkspace(n_t_int) : nothing
        
        if model == "nonlinear"
            St_avg = St_pred_mean_only!(nonlinear_ws, t_int, X_, Z_, results; thin=thin_factor)
        elseif model == "coxph"
            St_avg = coxph_St_pred_mean_only!(coxph_ws, t_int, X_, Z_, results; thin=thin_factor)
        end
        
        # Get KM values efficiently using pre-computed cache (batch lookup) - optimized
        G_Y_ = get_km(Y_)
        G_t_ = Vector{Float64}(undef, n_t_int)
        @inbounds @simd for k in 1:n_t_int
            t_val = t_int[k]
            if t_val <= 0
                G_t_[k] = 1.0
            elseif t_val >= max_time
                G_t_[k] = max_G
            else
                idx = searchsortedlast(all_possible_times, t_val)
                G_t_[k] = idx == 0 ? 1.0 : G_all[idx]
            end
        end
        
        # Optimized vectorized Brier score calculation - reduce branching
        brier_components = Vector{Float64}(undef, n_t_int)
        G_Y_safe = G_Y_ + 1e-5
        is_event = Delta_ == 1
        @inbounds @simd for k in 1:n_t_int
            t_k = t_int[k]
            G_t_safe = G_t_[k] + 1e-5
            if is_event && Y_ <= t_k
                # Event occurred before or at t_k
                St_k = St_avg[k]
                brier_components[k] = (St_k * St_k) / G_Y_safe
            elseif Y_ > t_k
                # Censored or event after t_k
                St_k = St_avg[k]
                one_minus_St = 1.0 - St_k
                brier_components[k] = (one_minus_St * one_minus_St) / G_t_safe
            else
                brier_components[k] = 0.0
            end
        end
        
        # Optimized trapezoidal integration for better accuracy
        # This is more accurate than simple mean when points are not uniformly spaced
        if n_t_int > 1
            # Optimized trapezoidal rule: avoid intermediate array allocation
            trap_sum = 0.0
            @inbounds @simd for k in 1:(n_t_int-1)
                trap_sum += (brier_components[k] + brier_components[k+1]) * 0.5 * (t_int[k+1] - t_int[k])
            end
            integrals[i_dag] = trap_sum / tau  # Normalize by tau
        else
            # Single point: use that value
            integrals[i_dag] = brier_components[1]
        end
        
        if show_progress
            next!(prog)
        end
    end
    
    integral = mean(integrals)
    return integral
end

end # module RJMCMCModel
