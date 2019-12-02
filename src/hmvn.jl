include("hchol.jl")
include("mvn.jl")

"""
    hmvn(Sigma, m, a1, b1, ns, N; tol = 1e-8, mu = 0)
    input: 
        - Sigma: covariance matrix
        - m: block size
        - a1: lower bound
        - b1: upper bound
        - ns: The number of sample size
        - N: Randomized QMC points
        - tol: tolerance
        - mu: mean
    output:
        - p_mean: estimated probabiliy
"""
function hmvn(Σ::Symmetric{T,Array{T,2}}, m::Int, a1::AbstractArray{T, 1}, b1::AbstractArray{T, 1}, 
    ns::Int, N::Int; tol = convert(T, 1e-8),
    μ::Array{T,1} = zeros(T, length(a1))) where T<:AbstractFloat

    n = size(Σ, 1) # total size
    (n % m == 0) || throw(ArgumentError("The condition m|n must be met."))
    (n >= m) || throw(ArgumentError("The condition n >= m must be met."))

    a = copy(a1)
    b = copy(b1)
    a -= μ # centering
    b -= μ # centering
    
    if (n == m)
        L = cholesky(Σ).L
        return mvn(L, a, b, ns, N, tol = tol)
    else
        B, UV = hchol(Σ, m)
        nb = length(B) # the number of blocks, r in Cao2019
        x = zeros(T, n) # record expectations
        log_P = 0.0

        for i in 1:nb
            j = (i-1) * m
            if i>1
                i1 = UV[i-1].i1
                j1 = UV[i-1].j1
                bsz = UV[i-1].bsz
                delta = zeros(T, bsz) # g in Cao2019
                if UV[i-1].rank != 0
                    delta[1:bsz] .= UV[i-1].U * (transpose(UV[i-1].V) * x[j1:j1+bsz-1] )
                end
                a[i1:i1+bsz-1] -= delta 
                b[i1:i1+bsz-1] -= delta 
            end
            ai = a[j+1:j+m]
            bi = b[j+1:j+m]
            pi = mvn(B[i], ai, bi, ns, N, tol = tol)
            xi = expt_tnorm(ai, bi, B[i], ns = ns, N = N)
            log_P += log(pi) # for numerical stability
            x[j+1:j+m] .= B[i] \ xi
        end

        return exp(log_P)
    end
    
end