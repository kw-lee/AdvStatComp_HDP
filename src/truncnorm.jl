using LinearAlgebra, Distributions, Statistics
include("mvn.jl")

#원래는 Quasi인데 일단 그냥 Monte-Carlo로
# 구현 완료 - 20191130 by kwlee at mvn.jl
function cdf_trunnormal(a::Array{Float64,1}, b::Array{Float64,1},μ::Array{Float64,1},Σ::Symmetric{Float64,Array{Float64,2}}; cnts = 10000)
    """
        calculate normal-cdf from a to b using **Monte-Carlo** simulation (Not quasi-MC: Todo)
    """
    d = length(a)
    temp = rand(MvNormal(μ,Σ), cnts) # generate samples
    ans = 0
    for i in 1:cnts
        if(sum(a .< sum(temp[:, i])) == d & sum(temp[:, i] .< b) == d)
            ans += 1 # count
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

    # Note (e_1, \cdots, e_d) = I_d
    μ + Σ * c / cdf_trunnormal(a, b, μ,Σ)
end