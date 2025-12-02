module RJMCMCModel

# Core RJMCMC samplers and survival utilities shared by simulation and real-data scripts.

using Random
using Distributions
using Statistics
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

# Piecewise log-baseline hazard evaluated at times t given knot locations taus and slopes gammas.
function loglambda_fun_est(t, tau, taus, gammas)
    taus_ = [0 taus' tau][1,:]
    H = size(taus)[1]
    if H==0
        return (gammas[1] .+ (gammas[2] .- gammas[1]) .* (t' .- taus_[1]) ./ (taus_[2] - taus_[1]))'
    else
    inds = taus_[2:(H+2)] .>= t' .> taus_[1:(H+1)]
    values = gammas[1:(H+1)] .+ (t' .- taus_[1:(H+1)]) .* 
            (gammas[2:(H+2)] .- gammas[1:(H+1)]) ./ 
            (taus_[2:(H+2)] - taus_[1:(H+1)])
    return sum(values .* inds, dims=1)[1,:]
    end
end

# Piecewise-linear nonlinear covariate effect g(z) over [a, b].
function g_fun_est(z, a, b, zetas, xis)
    zetas_ = [a zetas' b][1,:]
    K = size(zetas)[1]
    if K==0
        return (xis[1] .+ (xis[2] .- xis[1]) .* (z' .- zetas_[1]) ./ (zetas_[2] - zetas_[1]))'
    else
    inds = zetas_[2:(K+2)] .>= z' .> zetas_[1:(K+1)]
    values = xis[1:(K+1)] .+ (z' .- zetas_[1:(K+1)]) .* 
            (xis[2:(K+2)] .- xis[1:(K+1)]) ./ 
            (zetas_[2:(K+2)] - zetas_[1:(K+1)])
    return sum(values .* inds, dims=1)[1,:]
    end
end

# Integrated hazard Λ(t) corresponding to the piecewise log-λ definition.
function Lambda_fun_est(t, tau, taus, gammas)
    # Match historical formulation (model1229): clamp t to [0, tau] to avoid extrapolation.
    t = ifelse.(t .> tau, tau, t)
    taus_ = [0 taus' tau][1,:]
    H = size(taus)[1]
    lambda_mat = exp.(gammas[1:(H+1)] .+ (gammas[2:(H+2)] .- gammas[1:(H+1)]) .* 
                     ((t' .- taus_[1:(H+1)]) ./ (taus_[2:(H+2)] .- taus_[1:(H+1)])))
    part1 = (taus_[2:(H+2)] .- taus_[1:(H+1)]) ./ (gammas[2:(H+2)] .- gammas[1:(H+1)]) .* (lambda_mat .- exp.(gammas[1:(H+1)])) 
    part2 = [0 cumsum((taus_[2:(H+1)] .- taus_[1:(H)])' ./ (gammas[2:(H+1)] .- gammas[1:(H)])' .*
            (exp.(gammas[2:(H+1)]) .- exp.(gammas[1:(H)]))', dims=2)]'
    inds = taus_[2:(H+2)] .>= t' .> taus_[1:(H+1)]    
    return sum((part1 .+ part2) .* inds, dims=1)[1,:]
end

# Full log-likelihood for nonlinear model (baseline + covariates + nonlinear g(Z)).
function loglkh_cal(X,Z,Delta,Y, 
                    betas,
                    tau,taus,gammas,
                    a,b,zetas,xis)
    g_values = g_fun_est(Z, a, b, zetas, xis)
    loglambda_values = loglambda_fun_est(Y, tau, taus, gammas)
    Lambda_values = Lambda_fun_est(Y, tau, taus, gammas)
    Xbeta_values = X*betas
    loglkh = sum(Delta .* (loglambda_values .+ Xbeta_values .+ g_values) .- Lambda_values .* exp.(Xbeta_values .+ g_values))
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
        Hcan::Int=20,
        Kcan::Int=20
     )

    NS = ns
    BI = burn_in
    # ---------------------------- Hyper-Parameters ----------------------------
    # dimension of beta
    dim_beta = size(X,2)
    
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
	
    # prior: baseline hazard
    # candidate pool for taus
    taus_can = LinRange(0, tau, Hcan+2)[2:(Hcan+1)]
    # sample from candidate pool
    taus = H > 0 ? sort(sample(taus_can, H, replace=false)) : Float64[]
    taus_ = [0 taus' tau][1,:]
    
    gammas = zeros(H+2)
    gammas[1] = rand(Normal(0,5))
    for i in 2:(H+2)
        gammas[i] = rand(Normal(gammas[i-1],1))
    end  
    
    # prior: smooth function
    # candidate pool for zetas
    zetas_can = LinRange(a, b, Kcan+2)[2:(Kcan+1)]
    # sample from candidate pool
    zetas = K > 0 ? sort(sample(zetas_can, K, replace=false)) : Float64[]
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
    @time @showprogress for iter in 2:NS

        # ----------------------------- Update Gamma ----------------------------- 
        gammas_star = copy(gammas)
        agamma_vec = zeros(size(gammas)[1])
        sigma_gamma = sigmas_gamma_all[iter-1]
        
        for h in 1:(H+2)        
            # compute the denominator
            log_prob_de = log(pdf(Normal(0,5), gammas_star[1]))
            for hh in 2:(H+2)
                log_prob_de += log(pdf(Normal(gammas_star[hh-1], sigma_gamma), gammas_star[hh]))
            end
            log_de = log_prob_de + loglkh_cal(X,Z,Delta,Y, 
                                                betas,
                                                tau,taus,gammas_star,
                                                a,b,zetas,xis)
            
            # compute the numerator
            # propose a new gamma
            gammas_star[h] = rand(Uniform(gammas[h]-c_gamma, gammas[h]+c_gamma))
            
            log_prob_num = log(pdf(Normal(0,5), gammas_star[1]))                        
            for hh in 2:(H+2)
                log_prob_num += log(pdf(Normal(gammas_star[hh-1], sigma_gamma), gammas_star[hh]))
            end
            log_num = log_prob_num + loglkh_cal(X,Z,Delta,Y, 
                                                betas,
                                                tau,taus,gammas_star,
                                                a,b,zetas,xis)
            
            # acceptance ratio
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            # accept or not
            agamma_vec[h] = acc = rand() < acc_prob
            # update the gammas
            gammas_star[h] = acc * gammas_star[h] + (1-acc) * gammas[h]
        end
        
        # update the acceptance vector
        agammas_all[1:(H+2),iter] = agamma_vec
        
        # ----------------------------- Update Sigma_Gamma -----------------------------
        shape_param = 0.5 * H + 0.5
        scale_param = 0.5 * sum(diff(gammas_star).^2)
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
             log_de = loglkh_cal(X,Z,Delta,Y, 
                                        betas,
                                        tau,
                                        taus_star[2:(H+1)], 
                                        gammas_star,
                                        a,b,zetas,xis) +
                     　　　　log(taus_star[hc+2] - taus_star[hc+1]) + log(taus_star[hc+1] - taus_star[hc])
            
             log_num = loglkh_cal(X,Z,Delta,Y, 
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
                log_a_BM = loglkh_cal(X,Z,Delta,Y, 
                                        betas,
                                        tau,
                                        taus_star_add[2:(H_star+1)], 
                                        gammas_star_add,
                                        a,b,zetas,xis) - 
                           loglkh_cal(X,Z,Delta,Y, 
                                        betas,
                                        tau,
                                        taus_star[2:(H+1)], 
                                        gammas_star,
                                        a,b,zetas,xis) + 
                       log(2*H+3) + log(2*H+2) + log(tau_star-taus_star[h]) + log(taus_star[h+1]-tau_star) + 
                       log(pdf(Normal(gammas_star[h], sigma_gamma_star), gamma_star)) +
                       log(pdf(Normal(gamma_star, sigma_gamma_star), gammas_star[h+1])) -
                       2*log(tau) - log(taus_star[h+1] - taus_star[h]) - 
                       log(pdf(Normal(gammas_star[h], sigma_gamma_star), gammas_star[h+1])) +
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
                log_a_DM = loglkh_cal(X,Z,Delta,Y, 
                                    betas,
                                    tau,
                                    taus_star_rm[2:(H_star+1)], 
                                    gammas_star_rm,
                                    a,b,zetas,xis) - 
                        loglkh_cal(X,Z,Delta,Y, 
                                    betas,
                                    tau,
                                    taus_star[2:(H+1)], 
                                    gammas_star,
                                    a,b,zetas,xis) + 
                       2*log(tau) + log(taus_star_rm[h+1]-taus_star_rm[h]) + 
                       log(pdf(Normal(gammas_star_rm[h], sigma_gamma_star), gammas_star_rm[h+1])) - 
                       log(2*H+1) - log(2*H) - 
                       log(taus_star[h+2]-taus_star[h+1]) - log(taus_star[h+1]-taus_star[h]) - 
                       log(pdf(Normal(gammas_star[h+1], sigma_gamma_star), gammas_star[h+2])) -
                       log(pdf(Normal(gammas_star[h], sigma_gamma_star), gammas_star[h+1])) +
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
        
        for j in 1:dim_beta
            # compute the denominator
            log_de = loglkh_cal(X,Z,Delta,Y, 
                                betas_star,
                                tau,
                                taus, 
                                gammas,
                                a,b,zetas,xis)
            
            # compute the numerator
            # propose a new beta
            betas_star[j] = rand(Uniform(betas[j]-c_beta, betas[j]+c_beta))
    
            log_num = loglkh_cal(X,Z,Delta,Y, 
                                betas_star,
                                tau,
                                taus, 
                                gammas,
                                a,b,zetas,xis)
            
            # acceptance ratio
            aratio   = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            # accept or not
            abeta_vec[j] = acc = rand() < acc_prob
            # update the betas
            betas_star[j] = acc * betas_star[j] + (1-acc) * betas[j]
        end
        
        # update the coefficients
        betas_all[:,iter] = betas_star
        abetas_all[:,iter] = abeta_vec
        betas = betas_star
    
        # ----------------------------- Update Xi ----------------------------- 
        xis_star = copy(xis)
        axi_vec  = zeros(size(xis)[1])
        sigma_xi = sigmas_xi_all[iter-1]
    
        for k in 2:(K+2)        
            # compute the denominator
            log_prob_de = 0 
            for kk in 2:(K+2)
                log_prob_de += log(pdf(Normal(xis_star[kk-1],sigma_xi), xis_star[kk]))
            end
    
            log_de = log_prob_de +  loglkh_cal(X,Z,Delta,Y, 
                                                betas,
                                                tau,taus,gammas,
                                                a,b,zetas,xis_star)
            
            # compute the numerator
            # propose a new xi
            xis_star[k] = rand(Uniform(xis[k]-c_xi, xis[k]+c_xi))        
            log_prob_num = 0
            for kk in 2:(K+2)
                log_prob_num += log(pdf(Normal(xis_star[kk-1], sigma_xi), xis_star[kk]))
            end
            
            log_num = log_prob_num +  loglkh_cal(X,Z,Delta,Y, 
                                                betas,
                                                tau,taus,gammas,
                                                a,b,zetas,xis_star)
            
            # acceptance ratio
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            # accept or not
            axi_vec[k] = acc = rand() < acc_prob
            # update the xis
            xis_star[k] = acc * xis_star[k] + (1-acc) * xis[k]
        end
        
        # update the acceptance vector
        axis_all[1:(K+2),iter] = axi_vec
        
        # ----------------------------- Update Sigma_Xi -----------------------------
        shape_param = 0.5 * K + 0.5
        scale_param = 0.5 * sum(diff(xis_star).^2)
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
             log_de = loglkh_cal(X,Z,Delta,Y, 
								betas,
								tau,
								taus, 
								gammas,
								a,b,
								zetas_star[2:(K+1)],
								xis_star) +
                     　　　　log(zetas_star[kc+2] - zetas_star[kc+1]) + log(zetas_star[kc+1] - zetas_star[kc])
            
             log_num = loglkh_cal(X,Z,Delta,Y, 
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
                log_a_BM = loglkh_cal(X,Z,Delta,Y, 
                                     betas,
                                     tau,taus,gammas,
                                     a,b,zetas_star_add[2:(K_star+1)],
                                     xis_star_add) - 
                       loglkh_cal(X,Z,Delta,Y, 
                                     betas,
                                     tau,taus,gammas,
                                     a,b,zetas_star[2:(K+1)],
                                     xis_star) + 
                       log(2*K+3) + log(2*K+2) + log(zeta_star-zetas_star[k]) + log(zetas_star[k+1]-zeta_star) + 
                       log(pdf(Normal(xis_star[k], sigma_xi_star), xi_star)) +
                       log(pdf(Normal(xi_star, sigma_xi_star), xis_star[k+1])) - 
                       2*log(b-a) - log(zetas_star[k+1] - zetas_star[k]) - 
                       log(pdf(Normal(xis_star[k], sigma_xi_star), xis_star[k+1])) +
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
                log_a_DM = loglkh_cal(X,Z,Delta,Y, 
                                     betas,
                                     tau,taus,gammas,
                                     a,b,zetas_star_rm[2:(K_star+1)],
                                     xis_star_rm) - 
                           loglkh_cal(X,Z,Delta,Y, 
                                     betas,
                                     tau,taus,gammas,
                                     a,b,
                                     zetas_star[2:(K+1)],
                                     xis_star)  +
                           2*log(b-a) + log(zetas_star_rm[k+1]-zetas_star_rm[k]) + 
                           log(pdf(Normal(xis_star_rm[k], sigma_xi_star), xis_star_rm[k+1])) - 
                           log(2*K+1) - log(2*K) - 
                           log(zetas_star[k+2]-zetas_star[k+1]) - log(zetas_star[k+1]-zetas_star[k]) - 
                           log(pdf(Normal(xis_star[k+1], sigma_xi_star), xis_star[k+2])) -
                           log(pdf(Normal(xis_star[k], sigma_xi_star), xis_star[k+1])) +
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
        b_zeta::Float64=1.0    # Gamma rate for alpha_zeta
     )

    NS = ns
    BI = burn_in
    # ---------------------------- Hyper-Parameters ----------------------------
    dim_beta = size(X,2)
    
    # obtain the observed time
    obs_time = (Y .* Delta) 
    obs_time = obs_time[obs_time .> 0]
    tau  = maximum(obs_time)
    
    # prior parameter for number of knots (Poisson mean)
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
    
    # Initialize taus (empty for H=0)
    taus = Float64[]
    taus_ = [0.0; taus; tau]
    
    gammas = zeros(H+2)
    gammas[1] = rand(Normal(0,5))
    for i in 2:(H+2)
        gammas[i] = rand(Normal(gammas[i-1],1))
    end  
    
    # Initialize zetas (empty for K=0)
    zetas = Float64[]
    zetas_ = [a; zetas; b]
    
    xis = zeros(K+2)
    xis[1] = 0
    for i in 2:(K+2)
        xis[i] = rand(Normal(xis[i-1],1))
    end
    
    betas = zeros(dim_beta)
    
    # Store initial values
    H_all[1] = H
    gammas_all[1:(H+2),1] = gammas
    K_all[1] = K
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
    @time @showprogress for iter in 2:NS

        # ----------------------------- Update Gamma ----------------------------- 
        gammas_star = copy(gammas)
        agamma_vec = zeros(size(gammas)[1])
        sigma_gamma = sigmas_gamma_all[iter-1]
        
        for h in 1:(H+2)        
            log_prob_de = log(pdf(Normal(0,5), gammas_star[1]))
            for hh in 2:(H+2)
                log_prob_de += log(pdf(Normal(gammas_star[hh-1], sigma_gamma), gammas_star[hh]))
            end
            log_de = log_prob_de + loglkh_cal(X,Z,Delta,Y, betas, tau,taus,gammas_star, a,b,zetas,xis)
            
            gammas_star[h] = rand(Uniform(gammas[h]-c_gamma, gammas[h]+c_gamma))
            
            log_prob_num = log(pdf(Normal(0,5), gammas_star[1]))                        
            for hh in 2:(H+2)
                log_prob_num += log(pdf(Normal(gammas_star[hh-1], sigma_gamma), gammas_star[hh]))
            end
            log_num = log_prob_num + loglkh_cal(X,Z,Delta,Y, betas, tau,taus,gammas_star, a,b,zetas,xis)
            
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            agamma_vec[h] = acc = rand() < acc_prob
            gammas_star[h] = acc * gammas_star[h] + (1-acc) * gammas[h]
        end
        
        agammas_all[1:(H+2),iter] = agamma_vec
        
        # ----------------------------- Update Sigma_Gamma -----------------------------
        shape_param = 0.5 * H + 0.5
        scale_param = 0.5 * sum(diff(gammas_star).^2)
        sigma_gamma_star = sqrt(rand(InverseGamma(shape_param, scale_param)))
        sigmas_gamma_all[iter] = sigma_gamma_star
        
        # ----------------------------- Update Tau (with Dirichlet prior) -----------------------------
        taus_ = [0.0; taus; tau]
        taus_star = copy(taus_)
        taus_star_replace = copy(taus_)
                
        if H > 0
            hc = rand(1:H)
            tau_hc_star = rand(Uniform(taus_star[hc], taus_star[hc+2]))
            taus_star_replace[hc+1] = tau_hc_star
            
            # Log-likelihood ratio
            log_lik_de = loglkh_cal(X,Z,Delta,Y, betas, tau, taus_star[2:(H+1)], gammas_star, a,b,zetas,xis)
            log_lik_num = loglkh_cal(X,Z,Delta,Y, betas, tau, taus_star_replace[2:(H+1)], gammas_star, a,b,zetas,xis)
            
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
                
                log_a_BM = loglkh_cal(X,Z,Delta,Y, betas, tau, taus_star_add[2:(H_star+1)], gammas_star_add, a,b,zetas,xis) - 
                           loglkh_cal(X,Z,Delta,Y, betas, tau, taus_star[2:(H+1)], gammas_star, a,b,zetas,xis) +
                           log_prior_ratio +
                           log(pdf(Normal(gammas_star[h], sigma_gamma_star), gamma_star)) +
                           log(pdf(Normal(gamma_star, sigma_gamma_star), gammas_star[h+1])) -
                           log(pdf(Normal(gammas_star[h], sigma_gamma_star), gammas_star[h+1])) +
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
                
                log_a_DM = loglkh_cal(X,Z,Delta,Y, betas, tau, taus_star_rm[2:(H_star+1)], gammas_star_rm, a,b,zetas,xis) - 
                           loglkh_cal(X,Z,Delta,Y, betas, tau, taus_star[2:(H+1)], gammas_star, a,b,zetas,xis) +
                           log_prior_ratio +
                           log(pdf(Normal(gammas_star_rm[h], sigma_gamma_star), gammas_star_rm[h+1])) - 
                           log(pdf(Normal(gammas_star[h+1], sigma_gamma_star), gammas_star[h+2])) -
                           log(pdf(Normal(gammas_star[h], sigma_gamma_star), gammas_star[h+1])) +
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
        
        for j in 1:dim_beta
            log_de = loglkh_cal(X,Z,Delta,Y, betas_star, tau, taus, gammas, a,b,zetas,xis)
            betas_star[j] = rand(Uniform(betas[j]-c_beta, betas[j]+c_beta))
            log_num = loglkh_cal(X,Z,Delta,Y, betas_star, tau, taus, gammas, a,b,zetas,xis)
            
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            abeta_vec[j] = acc = rand() < acc_prob
            betas_star[j] = acc * betas_star[j] + (1-acc) * betas[j]
        end
        
        betas_all[:,iter] = betas_star
        abetas_all[:,iter] = abeta_vec
        betas = betas_star
    
        # ----------------------------- Update Xi ----------------------------- 
        xis_star = copy(xis)
        axi_vec = zeros(size(xis)[1])
        sigma_xi = sigmas_xi_all[iter-1]
    
        for k in 2:(K+2)        
            log_prob_de = 0 
            for kk in 2:(K+2)
                log_prob_de += log(pdf(Normal(xis_star[kk-1], sigma_xi), xis_star[kk]))
            end
            log_de = log_prob_de + loglkh_cal(X,Z,Delta,Y, betas, tau,taus,gammas, a,b,zetas,xis_star)
            
            xis_star[k] = rand(Uniform(xis[k]-c_xi, xis[k]+c_xi))        
            log_prob_num = 0
            for kk in 2:(K+2)
                log_prob_num += log(pdf(Normal(xis_star[kk-1], sigma_xi), xis_star[kk]))
            end
            log_num = log_prob_num + loglkh_cal(X,Z,Delta,Y, betas, tau,taus,gammas, a,b,zetas,xis_star)
            
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            axi_vec[k] = acc = rand() < acc_prob
            xis_star[k] = acc * xis_star[k] + (1-acc) * xis[k]
        end
        
        axis_all[1:(K+2),iter] = axi_vec
        
        # ----------------------------- Update Sigma_Xi -----------------------------
        shape_param = 0.5 * K + 0.5
        scale_param = 0.5 * sum(diff(xis_star).^2)
        sigma_xi_star = sqrt(rand(InverseGamma(shape_param, scale_param)))
        sigmas_xi_all[iter] = sigma_xi_star
        
        # ----------------------------- Update Zeta (with Dirichlet prior) -----------------------------
        zetas_ = [a; zetas; b]
        zetas_star = copy(zetas_)
        zetas_star_replace = copy(zetas_)
		
        if K > 0
            kc = rand(1:K)
            zeta_kc_star = rand(Uniform(zetas_star[kc], zetas_star[kc+2]))
            zetas_star_replace[kc+1] = zeta_kc_star
            
            # Log-likelihood ratio
            log_lik_de = loglkh_cal(X,Z,Delta,Y, betas, tau, taus, gammas, a,b, zetas_star[2:(K+1)], xis_star)
            log_lik_num = loglkh_cal(X,Z,Delta,Y, betas, tau, taus, gammas, a,b, zetas_star_replace[2:(K+1)], xis_star)
            
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
                
                log_a_BM = loglkh_cal(X,Z,Delta,Y, betas, tau,taus,gammas, a,b, zetas_star_add[2:(K_star+1)], xis_star_add) - 
                           loglkh_cal(X,Z,Delta,Y, betas, tau,taus,gammas, a,b, zetas_star[2:(K+1)], xis_star) +
                           log_prior_ratio +
                           log(pdf(Normal(xis_star[k], sigma_xi_star), xi_star)) +
                           log(pdf(Normal(xi_star, sigma_xi_star), xis_star[k+1])) - 
                           log(pdf(Normal(xis_star[k], sigma_xi_star), xis_star[k+1])) +
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
                
                log_a_DM = loglkh_cal(X,Z,Delta,Y, betas, tau,taus,gammas, a,b, zetas_star_rm[2:(K_star+1)], xis_star_rm) - 
                           loglkh_cal(X,Z,Delta,Y, betas, tau,taus,gammas, a,b, zetas_star[2:(K+1)], xis_star) +
                           log_prior_ratio +
                           log(pdf(Normal(xis_star_rm[k], sigma_xi_star), xis_star_rm[k+1])) - 
                           log(pdf(Normal(xis_star[k+1], sigma_xi_star), xis_star[k+2])) -
                           log(pdf(Normal(xis_star[k], sigma_xi_star), xis_star[k+1])) +
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

# logarithmic likelihood for CoxPH model (no nonlinear term here)
function coxph_loglkh_cal(X,Delta,Y, 
                            betas,
                            tau,taus,gammas)
    loglambda_values = loglambda_fun_est(Y, tau, taus, gammas)
    Lambda_values = Lambda_fun_est(Y, tau, taus, gammas)
    Xbeta_values = X*betas
    loglkh = sum(Delta .* (loglambda_values .+ Xbeta_values) .- Lambda_values .* exp.(Xbeta_values))
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
        Hcan::Int=20
     )

    NS = ns
    BI = burn_in

	X = hcat(X,Z)
    
    # ---------------------------- Hyper-Parameters ----------------------------    
    # Hmax
    Hmax = Hmax
    # dimension of beta
    dim_beta = size(X,2)
    
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
    
    # prior: baseline hazard
    # candidate pool for taus
    taus_can = LinRange(0, tau, Hcan+2)[2:(Hcan+1)]
    # sample from candidate pool
    taus = H > 0 ? sort(sample(taus_can, H, replace=false)) : Float64[]
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

    @time @showprogress for iter in 2:NS

        # ----------------------------- Update Gamma ----------------------------- 
        gammas_star = copy(gammas)
        agamma_vec = zeros(size(gammas)[1])
        sigma_gamma = sigmas_gamma_all[iter-1]
        
        for h in 1:(H+2)       
            # compute the denominator
            log_prob_de = log(pdf(Normal(0,5), gammas_star[1]))
            for hh in 2:(H+2)
                log_prob_de += log(pdf(Normal(gammas_star[hh-1], sigma_gamma), gammas_star[hh]))
            end
            log_de = log_prob_de + coxph_loglkh_cal(X,Delta,Y, 
                                                    betas,
                                                    tau,taus,gammas_star)
            
            # compute the numerator
            # propose a new gamma
            gammas_star[h] = rand(Uniform(gammas[h]-c_gamma, gammas[h]+c_gamma))
            log_prob_num = log(pdf(Normal(0,5), gammas_star[1]))
            for hh in 2:(H+2)
                log_prob_num += log(pdf(Normal(gammas_star[hh-1], sigma_gamma), gammas_star[hh]))
            end
            log_num = log_prob_num + coxph_loglkh_cal(X,Delta,Y, 
                                                    betas,
                                                    tau,taus,gammas_star)
            
            # acceptance ratio
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            # accept or not
            agamma_vec[h] = acc = rand() < acc_prob
            # update the gammas
            gammas_star[h] = acc * gammas_star[h] + (1-acc) * gammas[h]
        end
        
        # update the acceptance vector
        agammas_all[1:(H+2),iter] = agamma_vec
        
        # ----------------------------- Update Sigma_Gamma -----------------------------
        shape_param = 0.5 * H + 0.5
        scale_param = 0.5 * sum(diff(gammas_star).^2)
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
             log_de = coxph_loglkh_cal(X,Delta,Y,  
								betas,
								tau,
								taus_star[2:(H+1)], 
								gammas_star) +
                     　　　　log(taus_star[hc+2] - taus_star[hc+1]) + log(taus_star[hc+1] - taus_star[hc])
            
             log_num = coxph_loglkh_cal(X,Delta,Y, 
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
                log_a_BM = coxph_loglkh_cal(X,Delta,Y, 
                                            betas,
                                            tau,
                                            taus_star_add[2:(H_star+1)], 
                                            gammas_star_add) - 
                           coxph_loglkh_cal(X,Delta,Y, 
                                        betas,
                                        tau,
                                        taus_star[2:(H+1)], 
                                        gammas_star) + 
                       log(2*H+3) + log(2*H+2) + log(tau_star-taus_star[h]) + log(taus_star[h+1]-tau_star) + 
                       log(pdf(Normal(gammas_star[h], sigma_gamma_star), gamma_star)) +
                       log(pdf(Normal(gamma_star, sigma_gamma_star), gammas_star[h+1])) - 
                       2*log(tau) - log(taus_star[h+1] - taus_star[h]) - 
                       log(pdf(Normal(gammas_star[h], sigma_gamma_star), gammas_star[h+1])) +
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
                log_a_DM = coxph_loglkh_cal(X,Delta,Y, 
                                    betas,
                                    tau,
                                    taus_star_rm[2:(H_star+1)], 
                                    gammas_star_rm) - 
                        coxph_loglkh_cal(X,Delta,Y, 
                                    betas,
                                    tau,
                                    taus_star[2:(H+1)], 
                                    gammas_star) + 
                       2*log(tau) + log(taus_star_rm[h+1]-taus_star_rm[h]) + 
                       log(pdf(Normal(gammas_star_rm[h], sigma_gamma_star), gammas_star_rm[h+1])) - 
                       log(2*H+1) - log(2*H) - 
                       log(taus_star[h+2]-taus_star[h+1]) - log(taus_star[h+1]-taus_star[h]) - 
                       log(pdf(Normal(gammas_star[h+1], sigma_gamma_star), gammas_star[h+2])) -
                       log(pdf(Normal(gammas_star[h], sigma_gamma_star), gammas_star[h+1])) +
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
        
        for j in 1:dim_beta
            # compute the denominator
            log_de = coxph_loglkh_cal(X,Delta,Y, 
                                        betas_star,
                                        tau,
                                        taus, 
                                        gammas)
            
            # compute the numerator
            # propose a new beta
            betas_star[j] = rand(Uniform(betas[j]-c_beta, betas[j]+c_beta))
    
            log_num = coxph_loglkh_cal(X,Delta,Y, 
                                    betas_star,
                                    tau,
                                    taus, 
                                    gammas)
            
            # acceptance ratio
            aratio = exp(log_num - log_de)
            acc_prob = min(1, aratio)
            # accept or not
            abeta_vec[j] = acc = rand() < acc_prob
            # update the betas
            betas_star[j] = acc * betas_star[j] + (1-acc) * betas[j]
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

    St_all = zeros(size(t)[1], ns - bi)
    col_idx = 0
    for i in (bi+1):ns
        col_idx += 1
        Xbeta = sum(X_.*betas_all[:,i])
        K = Int(K_all[i])
        g = g_fun_est(Z_, a, b, zetas_all[1:K,i], xis_all[1:(K+2),i])[1]
        H = Int(H_all[i])
        Lambda = Lambda_fun_est(t, tau, taus_all[1:H,i], gammas_all[1:(H+2),i])
        St_all[:, col_idx] = exp.(-exp(Xbeta+g).*Lambda)
    end

    St_avg = mean(St_all, dims=2)
    St_lb = vquantile!(St_all, 0.025, dims=2)
    St_ub = vquantile!(St_all, 0.975, dims=2)
    
    return St_avg, St_lb, St_ub
end

# Survival prediction for CoxPH variant (linear Z term).
function coxph_St_pred(t,
                 X_, Z_,
                 results)

    betas_all, H_all, taus_all, gammas_all = results["betas"], results["H"], results["taus"], results["gammas"]
    tau = results["tau"]

    ns = get(results, "ns", size(betas_all, 2))
    bi = get(results, "burn_in", ns ÷ 2)

	d = size(X_)[1]
    St_all = zeros(size(t)[1], ns - bi)
    col_idx = 0
    for i in (bi+1):ns
        col_idx += 1
        Xbeta = sum(X_.*betas_all[1:d,i])
        H = Int(H_all[i])
        Lambda = Lambda_fun_est(t, tau, taus_all[1:H,i], gammas_all[1:(H+2),i])
        St_all[:, col_idx] = exp.(-exp(Xbeta+Z_*betas_all[d+1,i]).*Lambda)
    end

    St_avg = mean(St_all, dims=2)
    St_lb = vquantile!(St_all, 0.025, dims=2)
    St_ub = vquantile!(St_all, 0.975, dims=2)
    
    return St_avg, St_lb, St_ub
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

# Integrated Brier Score with inverse-probability-of-censoring weighting.
function IBS(Y_train, Delta_train, 
              X_train, Z_train, 
              Y_test, Delta_test,
              X_test, Z_test,
              results, model,
              n_int=200, 
              random_seed=2024)

    n_dag = size(Y_test)[1]  #size of the validating dataset
 
    Random.seed!(random_seed)
    integrals = zeros(n_dag)
    @showprogress for i_dag in 1:n_dag
        Y_, Delta_, X_, Z_ = Y_test[i_dag], Delta_test[i_dag], X_test[i_dag,:], Z_test[i_dag]
        tau = results["tau"]
        t_int = rand(Uniform(0,tau),n_int)
        if model == "nonlinear"
            St_avg, St_lb, St_ub = St_pred(t_int,X_,Z_,results)
        elseif model == "coxph"
            St_avg, St_lb, St_ub = coxph_St_pred(t_int,X_,Z_,results)
        end
        index1 = (Y_ .<= t_int) .* (Delta_==1)
        index2 = (Y_ .> t_int)
        G_Y_ = KM_est(Y_, Y_test, 1 .- Delta_test)
        G_t_ = [KM_est(t_int[i], Y_test, 1 .- Delta_test) for i in 1:n_int]
        integrals[i_dag] = mean((St_avg.^2 .*index1)./(G_Y_+1e-5) .+ ((1 .-St_avg).^2 .*index2)./(G_t_.+1e-5))
    end
    integral = mean(integrals)
    return integral
end

end # module RJMCMCModel
