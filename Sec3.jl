
using LinearAlgebra, Distributions, Statistics

function LDL(Σ::Symmetric{Float64,Array{Float64,2}}, m::Int, d::Int)#m is multiple of d
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
    (L, D)
end

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

#원래는 Quasi인데 일단 그냥 Monte-Carlo로
function cdf_trunnormal(a::Array{Float64,1}, b::Array{Float64,1},μ::Array{Float64,1},Σ::Symmetric{Float64,Array{Float64,2}}; cnts = 10000)
    d = length(a)
    temp = rand(MvNormal(μ,Σ), cnts)
    ans = 0
    for i in 1:cnts
        if(sum(a .< sum(temp[:, i])) == d & sum(temp[:, i] .< b) == d)
            ans += 1
        end
    end
    ans/cnts
end

function ex_trunnormal(a::Array{Float64,1}, b::Array{Float64,1}, μ::Array{Float64,1}, Σ::Symmetric{Float64,Array{Float64,2}})
    
    d = length(a)
    c = zeros(d)
    
    for l in 1:d
        μ1 = copy(μ[1:d .!= l] + Σ[1:d .!= l, l] * (a[l] - μ[l]) / Σ[l, l])
        μ2 = copy(μ[1:d .!= l] + Σ[1:d .!= l, l] * (b[l] - μ[l]) / Σ[l, l])
        Σl = copy(Symmetric(Σ[1:d .!= l, 1:d .!= l] - Σ[l, 1:d .!= l] * transpose(Σ[1:d .!= l, l]) / Σ[l, l]))
        c[l] = pdf(Normal(μ[l], sqrt(Σ[l, l])), a[l]) * cdf_trunnormal(a[1:d .!= l], b[1:d .!= l], μ1, Σl) - pdf(Normal(μ[l], sqrt(Σ[l, l])), b[l]) * cdf_trunnormal(a[1:d .!= l], b[1:d .!= l], μ2, Σl)
    end

    μ + Σ * c / cdf_trunnormal(a, b, μ,Σ)
end

function CMVN(Σ::Symmetric{Float64,Array{Float64,2}}, a::Array{Float64,1}, b::Array{Float64,1}, d::Int64, m::Int64)
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
        
        P *= cdf_trunnormal(a1, b1, zeros(d), copy(Symmetric(D1)))
        
        y[(j + 1):(j + d)] .= ex_trunnormal(a1, b1, zeros(d), copy(Symmetric(D1)))
    end
    P, y
end

function RCMVN(Σ::Symmetric{Float64,Array{Float64,2}}, a::Array{Float64,1}, b::Array{Float64,1}, d::Int64, m::Int64)
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
    CMVN(Symmetric(Σ), a, b, d, m)
end

iters = 250
ms = [16, 32, 64, 128]; ds = [1, 2, 4, 8, 16]
ans_1 = zeros(length(ms), length(ds)); time_1 = zeros(length(ms), length(ds))
ans_2 = zeros(length(ms), length(ds)); time_2 = zeros(length(ms), length(ds))

for i in 1:length(ms)
    for j in 1:length(ds)
        m = ms[i]; d = ds[j]
        for p in 1:iters
            Σ = Σ_generate(m)
            a = fill(-Inf, m)
            b = rand(Uniform(0, m), m)
            time_1[i, j] += @elapsed ans_1[i, j] += CMVN(Σ, a, b, d, m)[1]
            time_2[i, j] += @elapsed ans_2[i, j] += RCMVN(Σ, a, b, d, m)[1]
        end
    end
end
ans_1 ./= iters; time_1 ./= iters
ans_2 ./= iters; time_2 ./= iters

ans_1

time_1

ans_2

time_2
