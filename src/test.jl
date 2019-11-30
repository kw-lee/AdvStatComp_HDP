include("src/hchol.jl")

using LinearAlgebra
using Distributions

# test
A = rand(Normal(0,1), 8, 8)
A = A'A
m = 2

B, UV = hchol(A, m)
displayTree(UV)

L = uncompress(B, UV)
@assert A ≈ L*L'