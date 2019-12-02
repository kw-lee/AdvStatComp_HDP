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
function LDL(A::Symmetric{T,Array{T,2}}, d::Int; m::Int = size(Σ, 1)) where T<:AbstractFloat
    # m is multiple of d
    if m % d == 0
        Σ = Matrix(A)
        L = Matrix(1.0LinearAlgebra.I(m))
        D = Array{Matrix{T}}(undef, Int(m/d))
        for j = 1:Int(m/d)
            i = (j-1)*d + 1
            D[j] = Σ[i:(i + d - 1), i:(i + d - 1)]
            L[(i + d):m, i:(i + d - 1)] .= Σ[(i + d):m, i:(i + d - 1)] * inv(D[j])
            Σ[(i + d):m, (i + d):m] = Σ[(i + d):m, (i + d):m] .-
                L[(i + d):m, i:(i + d - 1)] * D[j] * transpose(L[(i + d):m, i:(i + d - 1)])
            # if(i + d < m)
            #     D[(i + d):m, (i + d):m] .= Σ[(i + d):m, (i + d):m]
            #     D[i:i+d-1, i+d:m] .= 0
            #     D[i+d:m, i:i+d-1] .= 0
            # end
        end
        L = LowerTriangular(L)
        return (L, D)
    end
end

"""
    Algorithm 2 in Cao2019 
    d-dimensional conditioning algorithm
"""
function CMVN(Σ::Symmetric{T,Array{T,2}}, a::Array{T,1}, b::Array{T,1}, d::Int;
    m::Int = size(Σ, 1), ns::Int = 10, N::Int = 1000, tol = convert(T, 1e-8)) where T<:AbstractFloat
    s = Int(m / d)
    y = zeros(m)
    P = 1.
    L, D = LDL(copy(Σ), d)
    for i in 1:s
        j = Int((i - 1) * d)
        g = L[(j + 1):(j + d), 1:j] * y[1:j]
        a1 = a[(j + 1):(j + d)] .- g
        b1 = b[(j + 1):(j + d)] .- g
        D1 = Symmetric(D[i])
        L1 = cholesky(D1).L
        # P *= cdf_trunnormal(a1, b1, zeros(d), copy(Symmetric(D1)))
        # y[(j + 1):(j + d)] .= ex_trunnormal(a1, b1, zeros(d), copy(Symmetric(D1)))
        # use ns=10 and N=1000 which is same in Genton2018
        p_i = mvn(L1, a1, b1, ns = ns, N = N, tol = tol)
        y_i = expt_tnorm(a1, b1, L1, ns = ns, N = N, tol = tol)

        P *= p_i
        y[(j + 1):(j + d)] .= y_i
    end
    return (P, y)
end


"""
    Cexpt_trnom(a, b, Sigma, d; m = size(L, 1), ns = 10, N = 1000, tol = 1e-8, mu = 0)
    function to calculate expectation of the sample from the truncated normal random variable
        using CMVN function
    input: 
        - a: lower bound
        - b: upper bound
        - Sigma: covariance matrix
        - d: d for CMVN
        - m: m for CMVN
        - ns: The number of sample size (defalut=10)
        - N: Randomized QMC points (defalt=1000)
        - tol: tolerance (defalt=1e-8)
        - mu: mean (default=0)
    output:
        - expectation
"""
function Cexpt_tnorm(a::AbstractArray{T,1}, b::AbstractArray{T,1}, Σ::Symmetric{T,Array{T,2}}, dCMVN::Int;
    m = size(Σ, 1), ns::Int = 10, N::Int = 1000, tol = convert(T, 1e-8), 
    μ::Array{T,1} = zeros(T, length(a))) where T<:AbstractFloat

    d = length(a)
    c = zeros(d)

    for l in 1:d
        μ1 = copy(μ[1:d .!= l] + Σ[1:d .!= l, l] * (a[l] - μ[l]) / Σ[l, l])
        μ2 = copy(μ[1:d .!= l] + Σ[1:d .!= l, l] * (b[l] - μ[l]) / Σ[l, l])
        Σl = copy(Symmetric(Σ[1:d .!= l, 1:d .!= l] - Σ[l, 1:d .!= l] * transpose(Σ[1:d .!= l, l]) / Σ[l, l]))
        c[l] = pdf(Normal(μ[l], sqrt(Σ[l, l])), a[l]) * 
            CMVN(Σl, a[1:d .!= l] - μ1, b[1:d .!= l] - μ1, dCMVN, ns = ns, N = N, tol = tol) - 
            pdf(Normal(0, sqrt(Σ[l, l])), b[l]) * 
            CMVN(Σl, a[1:d .!= l] - μ2, b[1:d .!= l] - μ2, dCMVN, ns = ns, N = N, tol = tol)
    end
    
    return (μ + Σ * c / CMVN(Σ, a, b, dCMVN, ns = ns, N = N, tol = tol))
end