using Primes
using Statistics
using Distributions
using StatsFuns

"""
    mvndns(n, N, L, x, a, b; tol)
    input:
        - n: dimension
        - N: Randomized QMC points
        - L: cholesky factor of the covariance matrix
        - a: lower bound
        - b: upper bound
    output:
        - p: estimated probabiliy
        - y: samples, Ly ~ truncated_normal(0, LL'; a, b)
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
            s = L[i,1:i]' * y[1:i, 1:N]
            s = s'
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
    c += x[n, :] .* dc
	c = map(x -> x > ALMOSTONE ? ALMOSTONE : x, c)
	buf = norminvcdf.(c) 
	y[n, :] .= buf 

    return (p, y)
end

"""
    mvn(L, a1, b1, v, e, ns, N)
    input: 
        - L: cholesky factor of the covariance matrix
        - a1: lower bound
        - b1: upper bound
        - ns: The number of sample size
        - N: Randomized QMC points
    output:
        - p_mean: estimated probabiliy
        - p_se: standard error
"""
function mvn(L::LowerTriangular{T,Array{T,2}}, a1::Vector{T}, b1::Vector{T}, 
    ns::Int, N::Int; tol = convert(T, 1e-8),
    μ::Array{T,1} = zeros(T, length(a1))) where T<:AbstractFloat

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
        p, y = mvndns(n, N, L, X, a, b, tol)
        values[i] = mean(p)
    end
    p_mean = mean(values) # estimated probabiliy

    return p_mean
end