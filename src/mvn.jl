using Primes
using Statistics
using Distributions
using StatsFuns

# """
#     mvndns(n, N, L, x, a, b; tol)
#     function to sample truncated normal probabilities
#     input:
#         - n: dimension
#         - N: Randomized QMC points
#         - L: cholesky factor of the covariance matrix
#         - a: lower bound
#         - b: upper bound
#     output:
#         - p: sampled probabilies
#         - ~~y: samples, Ly ~ truncated_normal(0, LL'; a, b)~~
# """
# function mvndns(n::Int, N::Int, L::LowerTriangular{T,Array{T,2}}, x::AbstractMatrix{T}, 
#     a::Matrix{T}, b::Matrix{T}, tol::T) where T<: AbstractFloat

#     ALMOSTONE = 1-tol
#     # c = Vector{T}(undef, N)
#     # d = Vector{T}(undef, N)
#     # dc = Vector{T}(undef, N)

#     y = zeros(T, n, N)
#     log_p = zeros(T, N) # numerical stability
#     s = zeros(T, N)

#     for i in 1:n 
#         if i > 1
#             c .+= x[i-1, :] .* dc
#             c = map(x -> x > ALMOSTONE ? ALMOSTONE : x, c)
#             buf = norminvcdf.(c)
#             y[i-1, :] .= buf
#             s = y[1:i-1, 1:N]' * L[1:i-1,i]
#         end

#         ct = L[i, i]
#         ai = fill(convert(T, -Inf), N)
#         bi = fill(convert(T, Inf), N)

#         for j in 1:N
#             if a[i,j] != -Inf
#                 ai[j] = a[i,j] - s[j]
#                 ai[j] /= ct
#             end
    
#             if b[i,j] != Inf
#                 bi[j] = b[i,j] - s[j]
#                 bi[j] /= ct
#             end
#         end
        
#         c = normcdf.(ai)
#         d = normcdf.(bi)
#         dc = d - c
#         log_p += log.(dc)
#     end
#     # c += x[n, :] .* dc
# 	# c = map(x -> x > ALMOSTONE ? ALMOSTONE : x, c)
# 	# buf = norminvcdf.(c) 
# 	# y[n, :] .= buf 

#     # return (p, y)
#     return exp.(log_p)
# end

"""
    mvn(L, a, b; ns = 10, N = 1000, tol = 1e-8, mu = 0)
    function to calculate normal probability P(a < x < b) where x ~ N(mu, LL^T) using 
        the method of Genz (1992), which relies on randomized quasi-Monte Carlo Richtmyer generators and tends to produce results that converge substantially faster than Monte Carlo points.
    input: 
        - L: cholesky factor of the covariance matrix
        - a: lower bound
        - b: upper bound
        - ns * N: sample size
        - ns: simulation size (defalut=10)
        - N: Randomized QMC points (defalt=1000)
        - tol: tolerance (defalt=1e-8)
        - mu: mean (default=0)
    output:
        - p_mean: estimated probabiliy
"""
function mvn(L::LowerTriangular{T,Array{T,2}}, a::AbstractArray{T, 1}, b::AbstractArray{T, 1};
    ns::Int = 10, N::Int = 1000, tol = convert(T, 1e-8),
    μ::Array{T,1} = zeros(T, length(a))) where T<:AbstractFloat

    a1 = copy(a)
    b1 = copy(b)
    n = size(L, 1)
    ALMOSTONE = 1-tol

    for i in 1:n
        if a1[i] != -Inf
            a1[i] = a1[i] - μ[i]
        end

        if b1[i] != Inf
            b1[i] = b1[i] - μ[i]
        end
    end

    if n == 1
        return normcdf.(b1)[1] - normcdf.(a1)[1]
    else
        # values produced by the ns samples, each with N randomized qmc points
        # i.e. total sample size = ns * N as in Genton et al. 2018.
        values = Vector{T}(undef, ns) 
        X = Matrix{T}(undef, n, N)
        # ax = reshape(repeat(a1, N), n, N)
        # bx = reshape(repeat(b1, N), n, N)

        # get prime numbers
        if n == 2
            prime_n = [2, 3]
        else
            prime_n = Primes.primes(Int(floor(5*n*log(n+1)/4)))
        end
        
        q = Vector{T}(undef, n)
        for i in 1:n
            q[i] = sqrt(prime_n[i])
        end

        for i in 1:ns
            xr = rand(T, n, 1) # xr ~ U(0,1)
            for j in 1:N
                X[:,j] = q * (1+j) + xr
            end
            X = map(x->abs(2*(x-floor(x))-1), X)

            samp = zeros(T, n, N)
            log_p = zeros(T, N) # numerical stability
            s = zeros(T, N)
            c = zeros(T, N)
            d = zeros(T, N)
            dc = zeros(T, N)

            for i in 1:n 
                if i > 1
                    c .+= X[i-1, :] .* dc
                    c = map(x -> x > ALMOSTONE ? ALMOSTONE : x, c)
                    buf = norminvcdf.(c)
                    samp[i-1, 1:N] .= buf
                    s = samp[1:i-1, 1:N]' * L[1:i-1,i]
                end
        
                ct = L[i, i]

                if a[i] != -Inf
                    ai = a[i] .- s
                    ai ./= ct
                else 
                    ai = fill(convert(T, -Inf), N)
                end

                if b[i] != Inf
                    bi = b[i] .- s
                    bi ./= ct
                else 
                    bi = fill(convert(T, Inf), N)
                end
                
                c = normcdf.(ai)
                d = normcdf.(bi)
                dc = d - c
                log_p += log.(dc)
            end

            p = exp.(log_p)

            values[i] = mean(filter(x -> !isnan(x), p)) # omit nan values
        end
        p_mean = mean(values) # estimated probabiliy

        return p_mean
    end
end

"""
    expt_trnom(a, b, L; ns = 10, N = 1000, tol = 1e-8, mu = 0)
    function to calculate expectation of the sample from the truncated normal random variable
    input: 
        - L: cholesky factor of the covariance matrix
        - a: lower bound
        - b: upper bound
        - ns: The number of sample size (defalut=10)
        - N: Randomized QMC points (defalt=1000)
        - tol: tolerance (defalt=1e-8)
        - mu: mean (default=0)
    output:
        - expectation
"""
function expt_tnorm(a::AbstractArray{T,1}, b::AbstractArray{T,1}, L::LowerTriangular{T,Array{T,2}};
    ns::Int = 10, N::Int = 1000, tol = convert(T, 1e-8), 
    μ::Array{T,1} = zeros(T, length(a))) where T<:AbstractFloat

    d = length(a)

    if d == 1
        # see https://en.wikipedia.org/wiki/Truncated_normal_distribution
        α = (a[1] - μ[1]) / L[1,1]
        β = (b[1] - μ[1]) / L[1,1]
        return (μ[1] + L[1,1] * (normpdf(α) - normpdf(β))/(normcdf(β) - normcdf(α)))
    else
        c = zeros(d)
        Σ = L * transpose(L)
        for l in 1:d
            μ1 = copy(μ[1:d .!= l] + Σ[1:d .!= l, l] * (a[l] - μ[l]) / Σ[l, l])
            μ2 = copy(μ[1:d .!= l] + Σ[1:d .!= l, l] * (b[l] - μ[l]) / Σ[l, l])
            Σl = copy(Symmetric(Σ[1:d .!= l, 1:d .!= l] - Σ[l, 1:d .!= l] * transpose(Σ[1:d .!= l, l]) / Σ[l, l]))
            Ll = cholesky(Σl).L
            c[l] = pdf(Normal(μ[l], sqrt(Σ[l, l])), a[l]) * 
                mvn(Ll, a[1:d .!= l], b[1:d .!= l], ns = ns, N = N, tol = tol, μ = μ1) - 
                pdf(Normal(0, sqrt(Σ[l, l])), b[l]) * 
                mvn(Ll, a[1:d .!= l], b[1:d .!= l], ns = ns, N = N, tol = tol, μ = μ2)
        end

        # Note (e_1, \cdots, e_d) = I_d
        return (μ + Σ * c / mvn(L, a, b, ns = ns, N = N, tol = tol))
    end
end