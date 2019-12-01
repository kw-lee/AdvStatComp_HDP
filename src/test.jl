using LinearAlgebra
using Distributions

# test for hchol
include("hchol.jl")
A = rand(Normal(0,1), 8, 8)
A = A'A
m = 2

B, UV = hchol(A, m)
displayTree(UV)

L = uncompress(B, UV)
@assert A ≈ L*L'

# test for mvn
include("mvn.jl")
dim = 8
a1 = repeat([-10.0], dim)
b1 = repeat([0.0], dim)
ns = 10
A = rand(Normal(0,1), dim, dim)
# A = convert(Matrix{Float64}, LinearAlgebra.I(dim))
Σ = A*A'
L = cholesky(Σ).L
N = 100
tol = 1e-8

## probs
include("truncnorm.jl")
prob1 = mvn(L, a1, b1, ns, N; tol = tol, μ = ones(Float64, length(a1)))
prob2 = cdf_trunnormal(a1, b1, ones(Float64, length(a1)), Symmetric(Σ))
prob1 = mvn(L, a1, b1, ns, N; tol = tol)
prob2 = cdf_trunnormal(a1, b1, zeros(Float64, length(a1)), Symmetric(Σ))

## truncated normal simulation
expt_exact = exp_truncnormal(a1, b1, Symmetric(Σ))

### p and y
n = dim
T = Float64
N = 1000
X = Matrix{T}(undef, n, N)
a = reshape(repeat(a1, N), n, N)
b = reshape(repeat(b1, N), n, N)
expt_mc = zeros(n)
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
xr = rand(T, n, 1) # xr ~ U(0,1)
for j in 1:N
    X[:,j] = q * (1+j) + xr
end
X = map(x->abs(2*(x-floor(x))-1), X)
p, y = mvndns(n, N, L, X, a, b, tol)
for j in 1:n
    expt_mc[j] = mean(y[j, :])
end
expt_mc = L * expt_mc


## test for hmvn
include("hmvn.jl")
m = 2
B, UV = hchol(Symmetric(Σ), m)

prob0 = hmvn(B, UV, a1, b1, ns, N; tol = tol, μ = ones(Float64, length(a1)))
prob1 = mvn(L, a1, b1, ns, N; tol = tol, μ = ones(Float64, length(a1)))
prob2 = cdf_trunnormal(a1, b1, ones(Float64, length(a1)), Symmetric(Σ))

prob0 = hmvn(B, UV, a1, b1, ns, N; tol = tol, μ = zeros(Float64, length(a1)) .- 2)
prob1 = mvn(L, a1, b1, ns, N; tol = tol, μ = zeros(Float64, length(a1)) .- 2)
prob2 = cdf_trunnormal(a1, b1, zeros(Float64, length(a1)) .- 2, Symmetric(Σ))