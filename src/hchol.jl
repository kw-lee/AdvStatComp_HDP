using LinearAlgebra
using Distributions
using LowRankApprox

mutable struct Tnode
    U::Matrix # U matrix
    V::Matrix # V matrix
    rank::Int # rank
    i1::Int  # row offset
    j1::Int # col offset
    bsz::Int # block size
end

#only n/m=2^k
function hchol_recur(A::Symmetric{T,Array{T,2}}, m::Int; tol=convert(T, 1e-8)) where T<: AbstractFloat
    A = Matrix(A)
    n = size(A)[1]
    nb = n ÷ 2
    if(n == m)
        return Matrix(cholesky(A).L)
    end
    
    A11 = copy(A[1:nb, 1:nb])
    A12 = copy(A[1:nb, (nb + 1):n])
    A21 = copy(A[(nb + 1):n, 1:nb])
    A22 = copy(A[(nb + 1):n, (nb + 1):n])    
    A[1:nb, 1:nb] = hchol_recur(Symmetric(A11), m)

    U12 = LowerTriangular(A[1:nb, 1:nb]) \ A12
    L21 = transpose(U12)
    # kmax denote max rank. better heuristic needed here
    kmax = 16 + Int(floor(3 * sqrt(nb)))
    kmax = kmax < nb ? kmax : nb
    F = svd(L21)
    A[(nb + 1):n, 1:nb] = F.U * diagm(vcat(sqrt.(F.S[1:kmax]), zeros(nb - kmax)))
    A[1:nb, (nb + 1):n] = F.V * diagm(vcat(sqrt.(F.S[1:kmax]), zeros(nb - kmax)))

    A[(nb + 1):n, (nb + 1):n] = hchol_recur(Symmetric(A22 - L21 * transpose(L21)), m)
    
    return A
end


function hchol(A::Symmetric{T,Array{T,2}}, m::Int; tol=convert(T, 1e-8)) where T<: AbstractFloat
    #setting for LowRankApprox
    opts = LRAOptions(maxdet_tol = tol, sketch_randn_niter=1)
    opts.sketch = :randn
    opts.rtol = 5*eps(T)
    
    A = Matrix(A)
    n = size(A)[1]
    nlev = Int(floor(log2(n/m)))
    UV = Array{Tnode}(undef, 2^nlev - 1)
    k = 1
    for i in 1:nlev
        bsz = n ÷ 2^i
        xbegin = 0; ybegin = bsz
        
        # kmax denote max rank. better heuristic needed here
        kmax = 8 + Int(floor(sqrt(bsz)))
        kmax = kmax < bsz ÷ 2 ? kmax : bsz ÷ 2
        for j in 1:2^(i - 1)
            if sum(abs, A[(xbegin + 1):(xbegin + bsz), (ybegin + 1):(ybegin + bsz)]) == 0
                UV[k] = Tnode(
                    zeros(bsz, 2), zeros(bsz, 2),
                    0, xbegin + 1, ybegin + 1, bsz)
            else
                U, S, V = psvd(A[(xbegin + 1):(xbegin + bsz), (ybegin + 1):(ybegin + bsz)], opts, rank = kmax)
                kmax = kmax < length(S) ? kmax : length(S)
                A[(xbegin + 1):(xbegin + bsz), (ybegin + 1):(ybegin + kmax)] = U * diagm(sqrt.(S)) 
                A[(xbegin + 1):(xbegin + bsz), (ybegin + kmax + 1):(ybegin + bsz)] .= 0 
                A[(ybegin + 1):(ybegin + bsz), (xbegin + 1):(xbegin + kmax)] = V * diagm(sqrt.(S)) 
                A[(ybegin + 1):(ybegin + bsz), (xbegin + kmax + 1):(xbegin + bsz)] .= 0
                
                UV[k] = Tnode(
                    A[(xbegin + 1):(xbegin + bsz), (ybegin + 1):(ybegin + kmax)],
                    A[(ybegin + 1):(ybegin + bsz), (xbegin + 1):(xbegin + kmax)],
                    kmax, xbegin + 1, ybegin + 1, bsz)
            end

            xbegin += bsz * 2; ybegin += bsz * 2
            k += 1            
        end
    end
    
    # bsz = n ÷ 2^nlev # block size; should be m
    xbegin = 0; ybegin = 0
    B = Vector{LowerTriangular{T,Array{T,2}}}(undef, 2^nlev)
    for j in 1:2^nlev
        L = A[(xbegin + 1):(xbegin + m), (ybegin + 1):(ybegin + m)]
        B[j] = cholesky(L).L
        # A[(xbegin + 1):(xbegin + m), (ybegin + 1):(ybegin + m)] = Matrix(B[j])
        xbegin += m; ybegin += m
    end
    
    return (B, UV)
end

# function uncompress!(A::Array{T,2}, m::Int) where T<: AbstractFloat
#     n = size(A)[1]
#     if(n == m)
#         return A = A * transpose(A)
#     end
#     nb = n ÷ 2
#     A[1:nb, 1:nb] = uncompress!(A[1:nb, 1:nb], m)
#     A[(nb + 1):n, 1:nb] = A[(nb + 1):n, 1:nb] * transpose(A[1:nb, (nb + 1):n])
#     A[1:nb, (nb + 1):n] = transpose(A[(nb + 1):n, 1:nb])
#     A[(nb + 1):n, (nb + 1):n] = uncompress!(A[(nb + 1):n, (nb + 1):n], m)
#     return A
# end

"""
    uncompress()
    generates the explicit matrix
"""
function uncompress(B::Array{LowerTriangular{T,Array{T,2}},1}, UV::Array{Tnode,1}) where T<: AbstractFloat
    
    nb = length(B)
    m = size(B[1], 1)
    A = zeros(T, nb*m, nb*m)
    for i in 1:nb
        A[(i-1)*m+1:i*m, (i-1)*m+1:i*m] = B[i] * B[i]'
    end

    for UV_i in UV
        A[UV_i.i1:UV_i.i1+UV_i.bsz-1, UV_i.j1:UV_i.j1+UV_i.bsz-1] .= UV_i.U * UV_i.V'
        A[UV_i.j1:UV_i.j1+UV_i.bsz-1, UV_i.i1:UV_i.i1+UV_i.bsz-1] = transpose(A[UV_i.i1:UV_i.i1+UV_i.bsz-1, UV_i.j1:UV_i.j1+UV_i.bsz-1])
    end

    # L = LowerTriangular(L)
    return A
end