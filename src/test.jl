using LinearAlgebra
using Distributions

# test for hchol
include("src/hchol.jl")
A = rand(Normal(0,1), 8, 8)
A = A'A
m = 2

B, UV = hchol(A, m)
displayTree(UV)

L = uncompress(B, UV)
@assert A ≈ L*L'

# test for mvn
include("src/mvn.jl")
dim = 5
a1 = repeat([-10.0], dim)
b1 = repeat([0.0], dim)
ns = 100
L = LinearAlgebra.I(dim)
L = convert(Matrix{Float64}, L)
N = 100
tol = 1e-8
prob, _ = mvn(L, a1, b1, ns, N; tol = tol)
@assert prob ≈ 1/(1<<dim)