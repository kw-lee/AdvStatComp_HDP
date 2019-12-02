using Primes
using Statistics
using Distributions
using StatsFuns

"""
    mvndns(n, N, L, x, a, b; tol)
    function to sample truncated normal probabilities
    input:
        - n: dimension
        - N: Randomized QMC points
        - L: cholesky factor of the covariance matrix
        - a: lower bound
        - b: upper bound
    output:
        - p: sampled probabiliy
        - ~~y: samples, Ly ~ truncated_normal(0, LL'; a, b)~~
"""
function mvndns(n::Int, N::Int, L::LowerTriangular{T,Array{T,2}}, x::AbstractMatrix{T}, 
    a::Matrix{T}, b::Matrix{T}, tol::T) where T<: AbstractFloat

    ALMOSTONE = 1-tol
    c = Vector{T}(undef, N)
    d = Vector{T}(undef, N)
    dc = Vector{T}(undef, N)

    y = zeros(T, n, N)
    p = ones(T, N)
	s = zeros(T, N)

    for i in 1:n 
        if i>1
            c .+= x[i-1, :] .* dc
            c = map(x -> x > ALMOSTONE ? ALMOSTONE : x, c)
            buf = norminvcdf.(c)
            y[i-1, :] .= buf
            s = y[1:i, 1:N]' * L[1:i,i]
        end
        ct = L[i, i]
        ai = a[i, :] .- s
        ai ./= ct
        bi = b[i, :] .- s
        bi ./= ct
        c = normcdf.(ai)
        d = normcdf.(bi)
        dc = d - c
        p = p .* dc
    end
    # c += x[n, :] .* dc
	# c = map(x -> x > ALMOSTONE ? ALMOSTONE : x, c)
	# buf = norminvcdf.(c) 
	# y[n, :] .= buf 

    # return (p, y)
    return p
end

"""
    mvn(L, a, b, ns, N; tol = 1e-8, mu = 0)
    function to calculate normal probability P(a < x < b) where x ~ N(mu, LL^T) using 
        the method of Genz (1992), which relies on randomized quasi-Monte Carlo Richtmyer generators and tends 
        to produce results that converge substantially faster than Monte Carlo points.
    input: 
        - L: cholesky factor of the covariance matrix
        - a: lower bound
        - b: upper bound
        - ns: The number of sample size
        - N: Randomized QMC points
        - tol: tolerance
        - mu: mean
    output:
        - p_mean: estimated probabiliy
"""
function mvn(L::LowerTriangular{T,Array{T,2}}, a::AbstractArray{T, 1}, b::AbstractArray{T, 1}, 
    ns::Int, N::Int; tol = convert(T, 1e-8),
    μ::Array{T,1} = zeros(T, length(a))) where T<:AbstractFloat

    a1 = copy(a)
    b1 = copy(b)
    a1 -= μ # centering
    b1 -= μ # centering

    # values produced by the ns samples, each with N randomized qmc points
    values = Vector{T}(undef, ns) 
    n = size(L, 1)
    X = Matrix{T}(undef, n, N)
    a = reshape(repeat(a1, N), n, N)
    b = reshape(repeat(b1, N), n, N)

    # get prime numbers
    if n == 1
        prime_n = 2
    elseif n == 2
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
        p = mvndns(n, N, L, X, a, b, tol)
        values[i] = mean(filter(x -> !isnan(x), p)) # omit nan values
    end
    p_mean = mean(values) # estimated probabiliy

    return p_mean
end

"""
    expt_trnom(a, b, L; ns = 10, N = 1000, tol = 1e-8, mu = 0)
    function to calculate expectation of the sample from the truncated normal random variable
    input: 
        - L: cholesky factor of the covariance matrix
        - a: lower bound
        - b: upper bound
        - ns: The number of sample size
        - N: Randomized QMC points
        - tol: tolerance
        - mu: mean
    output:
        - expectation
"""
function expt_tnorm(a::AbstractArray{T,1}, b::AbstractArray{T,1}, L::LowerTriangular{T,Array{T,2}};
    ns = 10, N = 1000, tol = convert(T, 1e-8), μ::Array{T,1} = zeros(T, length(a))) where T<:AbstractFloat

    d = length(a)
    c = zeros(d)
    Σ = L * transpose(L)

    for l in 1:d
        μ1 = copy(μ[1:d .!= l] + Σ[1:d .!= l, l] * (a[l] - μ[l]) / Σ[l, l])
        μ2 = copy(μ[1:d .!= l] + Σ[1:d .!= l, l] * (b[l] - μ[l]) / Σ[l, l])
        Σl = copy(Symmetric(Σ[1:d .!= l, 1:d .!= l] - Σ[l, 1:d .!= l] * transpose(Σ[1:d .!= l, l]) / Σ[l, l]))
        Ll = cholesky(Σl).L
        c[l] = pdf(Normal(μ[l], sqrt(Σ[l, l])), a[l]) * 
            mvn(Ll, a[1:d .!= l], b[1:d .!= l], ns, N, tol = tol, μ = μ1) - 
            pdf(Normal(0, sqrt(Σ[l, l])), b[l]) * 
            mvn(Ll, a[1:d .!= l], b[1:d .!= l], ns, N, tol = tol, μ = μ2)
    end

    # Note (e_1, \cdots, e_d) = I_d
    return (μ + Σ * c / mvn(L, a, b, ns, N, tol = tol))
end