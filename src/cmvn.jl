using LinearAlgebra, Distributions, Statistics, SparseArrays

include("generate.jl")
# include("truncnorm.jl")
include("mvn.jl")

"""
    LDL(A, d; m = size(Sigma, 1))
    Algorithm 1 in Cao2019
    LDL decomposition: A = L * Matrix(D) * L'
    input:
        - A: positive definite matrix
        - d: block size
        - m: size of the matrices
    output:
        - L: LowerTriangular
        - D: array{Matrix}
"""
function LDL(A::Symmetric{T,Array{T,2}}, d::Int; m::Int = size(A, 1)) where T<:AbstractFloat
    # m is multiple of d
    (m % d == 0) || throw(ArgumentError("The condition d|m must be met."))
    Σ = Matrix(A)
    L = Matrix(1.0LinearAlgebra.I(m))
    D = Array{Matrix{T}}(undef, Int(m/d))
    for j = 1:Int(m/d)
        i = (j-1)*d + 1
        D[j] = Σ[i:(i + d - 1), i:(i + d - 1)]
        L[(i + d):m, i:(i + d - 1)] .= Σ[(i + d):m, i:(i + d - 1)] * inv(D[j])
        Σ[(i + d):m, (i + d):m] = Σ[(i + d):m, (i + d):m] .-
            L[(i + d):m, i:(i + d - 1)] * D[j] * transpose(L[(i + d):m, i:(i + d - 1)])
    end
    L = LowerTriangular(L)
    return (L, D)
end

"""
    CMVN(Sigma, a, b, d; m = size(Sigma, 1), ns = 10, N = 1000, tol = 1e-8)
    Algorithm 2 in Cao2019; d-dimensional conditioning algorithm
    input: 
        - Sigma: covariance matrix
        - a: lower bound
        - b: upper bound
        - d: d for CMVN
        - m: m for CMVN
        - ns * N: sample size
        - ns: simulation size (defalut=10)
        - N: Randomized QMC points (defalt=1000)
        - tol: tolerance (defalt=1e-8)
    output:
        - P: estimated probabiliy
        - y: truncated expectation
"""
function CMVN(Σ::Symmetric{T,Array{T,2}}, a::Array{T,1}, b::Array{T,1}, d::Int;
    m::Int = size(Σ, 1), ns::Int = 10, N::Int = 1000, tol = convert(T, 1e-8)) where T<:AbstractFloat
    (m % d == 0) || throw(ArgumentError("The condition d|m must be met."))
    s = trunc(Int, m / d)
    y = zeros(m)
    P = 1.
    L, D = LDL(copy(Σ), d)
    for i in 1:s
        j = Int((i - 1) * d)
        g = copy(L[(j + 1):(j + d), 1:j] * y[1:j])
        a1 = a[(j + 1):(j + d)] .- g
        b1 = b[(j + 1):(j + d)] .- g
        D1 = Symmetric(copy(D[i]))
        L1 = cholesky(D1).L
        # P *= cdf_trunnormal(a1, b1, zeros(d), copy(Symmetric(D1)))
        # y[(j + 1):(j + d)] .= ex_trunnormal(a1, b1, zeros(d), copy(Symmetric(D1)))
        p_i = mvn(L1, a1, b1, ns = ns, N = N, tol = tol)
        y_i = expt_tnorm(a1, b1, L1, ns = ns, N = N, tol = tol)

        P *= p_i
        y[(j + 1):(j + d)] .= y_i
        
        #println("############", i)
        #println(P)
        #println(a1)
        #println(b1)
        #println(L1)
        #println(expt_tnorm(a1, b1, L1, ns = ns, N = N, tol = tol))
        #println(a1)
        #println(b1)
        #println(L1)
    end
    return (P, y)
end