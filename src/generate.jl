using LinearAlgebra, Distributions, Statistics

function haar_generate(m::Int)
    X = rand(Normal(0, 1), m, m)
    Q = Matrix(1.0I, m, m)
    D = zeros(m)
    for i in m:-1:2
        u = X[:, 1]
        u[1] -= norm(u, 2)
        P = Matrix(1.0I, i, i) - 2.0 .* u * transpose(u) ./ (norm(u, 2)^2)
        D[m - i + 1] = sign((P * X)[1, 1])
        X = (P * X)[2:i, 2:i]

        temp = Matrix(1.0I, m, m)
        temp[(m - i + 1):m, (m - i + 1):m] .= P
        Q = Q * temp
    end
    D[m] = sign(X[1, 1])
    diagm(0 => D) * Q
end

function Σ_generate(m::Int)
    #Data generate
    #Sigma generate from Haar distribution over the orthogonal matrix gruop
    Q = haar_generate(m)

    #J : a diagonal matrix with die diagonal coefficients independently drawn from U(0,1)
    J = zeros(m, m)
    J[diagind(J)] .= rand(Uniform(0, 1), m)
    J
    
    Symmetric(Q * J * transpose(Q))
end

function Σ_const_generate(n::Int, θ::T) where T <: AbstractFloat
    Σ_const = (1-θ)*LinearAlgebra.I(n) + θ*ones(n, n)
    Σ_const = Symmetric(Σ_const)
    return Σ_const
end

function Σ_1d_generate(n::Int, β::T) where T>: AbstractFloat
    Σ_1d = ones(n, n)
    for i in 1:n
        for j in 1:i
            d_ij = abs(i - j)
            Σ_1d[i, j] = exp(-d_ij / β)
            Σ_1d[j, i] = exp(-d_ij / β)    
        end
    end
    Σ_1d = Symmetric(Σ_1d)
    return Σ_1d
end

function morton(n::Int64)
    if n == 1
        return [1], [1]
    end
    a, b = morton(n÷2)
    return vcat(a, a .+ n÷2, a, a .+ n÷2), vcat(b, b, b .+ n÷2, b .+ n÷2)
end
"""
Exponential Covariance Matrix with β
n points evenly distributed on a grid in the unit square and indexed with Morton's order
"""
function Σ_morton_generate(n::Int, β::T) where T>: AbstractFloat
    a, b = morton(n)7
    for i in 1:N
        for j in 1:N
            Σ[i, j] = exp( -norm([a[i] - a[j], b[i] - b[j]], 2) / β)
        end
    end
    Σ = Symmetric(Σ)
    return Σ
end


