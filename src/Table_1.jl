include("hchol.jl")
β = 0.3
ns = [16, 32, 64, 128]
chol_time = zeros(4)
potrf_time = zeros(4)
hchol_time = zeros(4)
for i in 1:1
    n = ns[i]
    N = n * n
    Σ = zeros(N, N)
    #Exponential Covariance Matrix with β
    #n points evenly distributed on a grid in the unit square and indexed with Morton's order
    function morton(n::Int64)
        if n == 1
            return [1], [1]
        end
        a, b = morton(n÷2)
        return vcat(a, a .+ n÷2, a, a .+ n÷2), vcat(b, b, b .+ n÷2, b .+ n÷2)
    end

    a, b = morton(n)
    for i in 1:N
        for j in 1:N
            Σ[i, j] = exp( -norm([a[i] - a[j], b[i] - b[j]], 2) / β)
        end
    end
    Σ = Symmetric(Σ)
    chol_time[i] = @elapsed cholesky(Σ)
    potrf_time[i] = @elapsed LAPACK.potrf!('L', Matrix(Σ))[1]
    hchol_time[i] = @elapsed hchol(Σ, 1)
    
    println(n)
    println(chol_time[i])
    println(potrf_time[i])
    println(hchol_time[i])
end