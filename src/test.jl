using LinearAlgebra
using Distributions

cd("./src")

include("hchol.jl")
include("mvn.jl")
include("truncnorm.jl")
include("hcmvn.jl")

# test for hchol
A = rand(Normal(0,1), 8, 8)
A = A'A
A = Symmetric(A)
m = 2
B, UV = hchol(A, m)
# displayTree(UV)

L = uncompress(B, UV)
@assert A ≈ L*L'

# test for mvns

dim = 16
m = 4
a1 = repeat([-10.0], dim)
b1 = repeat([0.0], dim)
ns = 10

## 1. Sigma = I
A = convert(Matrix{Float64}, LinearAlgebra.I(dim))
Σ = A*A'
L = cholesky(Σ).L
N = 1000
tol = 1e-8

### test for expt_tnorm
expt_tnorm([0.0, 0.0, 0.0], [0.5, 1.0, 1.5], LowerTriangular([1.0 0 0 ; 0 1.0 0 ; 0 0 1.0]))
expt_tnorm([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0], LowerTriangular([1.0 0 0 ; 0 1.0 0 ; 0 0 1.0]))

### test for probs
prob0 = HCMVN(Symmetric(Σ), m, a1, b1, ns, N; tol = tol, μ = ones(Float64, length(a1)))
prob1 = mvn(L, a1, b1, ns = ns, N = N, tol = tol, μ = ones(Float64, length(a1)))
prob2 = cdf_trunnormal(a1, b1, ones(Float64, length(a1)), Symmetric(Σ))
prob0 = HCMVN(Symmetric(Σ), m, a1, b1, ns, N; tol = tol, μ = b1)
prob1 = mvn(L, a1, b1, ns = ns, N = N, tol = tol, μ = b1)
prob2 = cdf_trunnormal(a1, b1, zeros(Float64, length(a1)), Symmetric(Σ))

## 2. Sigma = AA', A_ij ~iid N(0,1)
A = rand(Normal(0,1), dim, dim)
Σ = A*A'
L = cholesky(Σ).L
N = 1000
tol = 1e-8

prob0 = HMVN(a1, b1, Symmetric(Σ), m, ns = ns, N = N, tol = tol, μ = b1 .- 5)
prob1 = mvn(L, a1, b1, ns = ns, N = N, tol = tol, μ = b1 .- 5)
prob2 = cdf_trunnormal(a1, b1, b1 .- 5, Symmetric(Σ))

prob0 = HMVN(a1, b1, Symmetric(Σ), m, ns = ns, N = N, tol = tol, μ = ones(Float64, length(a1)))
prob1 = mvn(L, a1, b1, ns = ns, N = N, tol = tol, μ = ones(Float64, length(a1)))
prob2 = cdf_trunnormal(a1, b1, ones(Float64, length(a1)), Symmetric(Σ))