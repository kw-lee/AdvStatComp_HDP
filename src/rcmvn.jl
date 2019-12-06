using LinearAlgebra, Distributions, Statistics

include("cmvn.jl")

"""
Algorithm 3 in Cao2019
d-dimensional conditioning algorithm with univariate reordering
"""
function RCMVN(Σ::Symmetric{T,Array{T,2}}, a::Array{T,1}, b::Array{T,1}, d::Int;
    m::Int = size(Σ, 1), ns::Int = 10, N::Int = 1000, tol = convert(T, 1e-8)) where T<:AbstractFloat
    (m % d == 0) || throw(ArgumentError("The condition d|m must be met."))
    
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
    return CMVN(Symmetric(Σ), a, b, d, m = m, ns = ns, N = N, tol = tol)
end