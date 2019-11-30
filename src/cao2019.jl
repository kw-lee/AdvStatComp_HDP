using LinearAlgebra, Distributions, Statistics

include("generate.jl")
include("truncnorm.jl")
include("mvn.jl")

function LDL(Σ::Symmetric{Float64,Array{Float64,2}}, m::Int, d::Int)
    """
        Algorithm 1 in Cao2019
        d-dimensional conditioning algorithm
    """
    # m is multiple of d
    if m % d == 0
        Σ = Matrix(Σ)
        L = Matrix(1.0I, m, m)
        D = zeros(m, m)
        for i = 1:d:m-d+1
            D[i:(i + d - 1), i:(i + d - 1)] .= Σ[i:(i + d - 1), i:(i + d - 1)]
            L[(i + d):m, i:(i + d - 1)] .= Σ[(i + d):m, i:(i + d - 1)] * inv(D[i:(i + d - 1), i:(i + d - 1)])
            Σ[(i + d):m, (i + d):m] = Σ[(i + d):m, (i + d):m] .-
                L[(i + d):m, i:(i + d - 1)] * D[i:(i + d - 1), i:(i + d - 1)] * transpose(L[(i + d):m, i:(i + d - 1)])
            if(i + d < m)
                D[(i + d):m, (i + d):m] .= Σ[(i + d):m, (i + d):m] 
            end
        end
        return (L, D)
    end
end

function CMVN(Σ::Symmetric{Float64,Array{Float64,2}}, a::Array{Float64,1}, b::Array{Float64,1}, d::Int64, m::Int64)
    """
        Algorithm 2 in Cao2019 
    """
    s = Int(m / d)
    y = zeros(m)
    P = 1.
    L, D = LDL(copy(Σ), m, d)
    for i in 1:s
        j = Int((i - 1) * d)
        g = L[(j + 1):(j + d), 1:j] * y[1:j]
        a1 = a[(j + 1):(j + d)] .- g
        b1 = b[(j + 1):(j + d)] .- g
        D1 = D[(j + 1):(j + d), (j + 1):(j + d)]
        
        L1 = chol(D1).L
        # P *= cdf_trunnormal(a1, b1, zeros(d), copy(Symmetric(D1)))
        # y[(j + 1):(j + d)] .= ex_trunnormal(a1, b1, zeros(d), copy(Symmetric(D1)))
        # use ns=10 and N=1000 which is same in Genton2018
        (p_i, y_i) = mvn(L1, a1, b1, 10, 1000)

        P *= p_i
        y[(j + 1):(j + d)] = y_i[1:end]
    end
    return (P, y)
end

function RCMVN(Σ::Symmetric{Float64,Array{Float64,2}}, a::Array{Float64,1}, b::Array{Float64,1}, d::Int64, m::Int64)
    """
        Algorithm 3 in Cao2019
        d-dimensional conditioning algorithm with univariate reordering
    """
    Σ = Matrix(Σ)
    y = zeros(m)
    C = copy(Σ)
    a_prime = b_prime = 0
    
    for i in 1:m
        if i > 1
            y[i - 1] = (pdf(Normal(), a_prime) - pdf(Normal(), b_prime)) / (cdf(Normal(), b_prime) - cdf(Normal(), a_prime))
        end
        
        min_temp = 10; j_min = i
        
        for j in i:m
            temp1 = (b[j] - transpose(C[j, 1:(i - 1)]) * y[1:(i - 1)]) / (sqrt(Σ[j, j] - transpose(C[j, 1:(i - 1)]) * C[j, 1:(i - 1)]))
            temp2 = (a[j] - transpose(C[j, 1:(i - 1)]) * y[1:(i - 1)]) / (sqrt(Σ[j, j] - transpose(C[j, 1:(i - 1)]) * C[j, 1:(i - 1)]))
            temp = cdf(Normal(), temp1) - cdf(Normal(), temp2)
            if(min_temp > temp)
                j_min = j
                temp = min_temp
            end
        end
        
        j = j_min
        Σ[:, [i, j]] = Σ[:, [j, i]]; Σ[[i, j], :] = Σ[[j, i], :]
        C[:, [i, j]] .= C[:, [j, i]]; C[[i, j], :] .= C[[j, i], :]
        a[[i, j]] .= a[[j, i]]; b[[i, j]] .= b[[j, i]]
        C[i, i] = sqrt(Σ[i, i] - transpose(C[i, 1:(i - 1)]) * C[i, 1:(i - 1)])
        for j in  (i + 1):m
            C[j, i] = (Σ[j, i] - transpose(C[i, 1:(i - 1)]) * C[j, 1:(i - 1)]) / C[i, i]
        end
        a_prime = (a[i] - transpose(C[i, 1:(i - 1)]) * y[1:(i - 1)]) / C[i, i]
        b_prime = (b[i] - transpose(C[i, 1:(i - 1)]) * y[1:(i - 1)]) / C[i, i]
        
    end
    return CMVN(Symmetric(Σ), a, b, d, m)
end