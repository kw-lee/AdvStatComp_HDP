"""
    Julia implementation for Genton, M. G., Keyes, D. E., & Turkiyyah, G. (2018). Hierarchical decompositions for the computation of high-dimensional multivariate normal probabilities. Journal of Computational and Graphical Statistics, 27(2), 268-277.
    You can download original C++ codes at https://amstat.tandfonline.com/doi/abs/10.1080/10618600.2017.1375936
"""

using LinearAlgebra
using Distributions

mutable struct Tnode
    U::Matrix # U matrix
    V::Matrix # V matrix
    rank::Int # rank
    i1::Int  # row offset
    j1::Int # col offset
    bsz::Int # block size
    q::Int # hierarchical level
end

"""
    function to perform low-rank approximation A ~ UV'
    update U and V
"""
function updateTree!(UV::Tnode, A::SubArray{T, 2}, tol) where T<:AbstractFloat
    
    n = size(A, 1)
    # kmax denote max rank. better heuristic needed here
    kmax = 16 + Int(floor(3 * sqrt(n)))
    kmax = kmax < n ? kmax : n; 

    # For numerical stability
    # Let AZ = QR, AO = A'Q = USV' for Z ~ N(0,1) 
    # A = QQ' A = QVSU' ~ QV[1:k] S[1:k] U[1:k]'
    Omega = rand(Normal(0,1), n, kmax)
    AO = A * Omega
    Q, R = qr(AO)

    AO = transpose(A) * Q
    F = svd(AO)
    Uf = F.U
    Vf = F.V
    s = F.S

    rank = kmax
    for k in 1:kmax
        if s[k] <= tol
            rank = k-1
            break
        end
    end

    if (rank>1)&(rank==kmax) 
        println("Warning: max rank reached")
    end

    s2 = sqrt.(s[1:rank])
    U2 = Matrix{T}(undef, n, rank)
    V2 = Matrix{T}(undef, kmax, rank)
    for i in 1:n
        for j in 1:rank
            U2[i,j] = Uf[i,j] * s2[j]
        end
    end

    for i in 1:kmax
        for j in 1:rank
            V2[i,j] = Vf[i,j] * s2[j]
        end
    end

    UV.rank = rank
    UV.U = Q*V2
    UV.V = U2
end

"""
    build Tree
    input: n, nlev
    output: Array{Tnode}
    todo: split size 2 to general integer d (Cao2019)
"""
function buildTree(T::Type, n::Int, nlev::Int)
    tree = Array{Tnode}(undef, (1<<nlev) - 1)
    for q in 1:nlev
        bsz = n ÷ (1<<q)  # n / 2^(q+1), size of block at level q
        for i in 1:(1<<(q-1))
            zi = Int(i + (1<<(q-1)) - 1) # index of block in binary tree
            tree[zi] = Tnode(
                Matrix(undef, bsz, 2), Matrix(undef, 2, bsz), 2, # U, V, rank are undefined
                (2*i-1)*bsz+1, 2*(i-1)*bsz+1, bsz, q)
        end
    end
    return tree
end

"""
    Build a dense Cholesky decomposition and put it in Hierarchical matrix format
    todo: split size 2 to general integer d (Cao2019)
"""
function hchol(A::Matrix{T}, m::Int; tol=convert(T, 1e-8)) where T<: AbstractFloat
    n = size(A, 1)
    nb = n ÷ m # only for m|n
    nlev = Int(floor(log2(n/m)))
    if m>n 
        return 1
    end

    L = cholesky(A).L

    B = Vector{LowerTriangular{T,Array{T,2}}}(undef, nb)
    for i in 1:nb
        # diagonal matrix B
        B[i] = LowerTriangular(L[(i-1)*m+1:i*m, (i-1)*m+1:i*m])
    end

    if nlev>0
        # order = inorder(0, nlev)
        UV = buildTree(T, n, nlev)
        # ordering
        # UV = permuteTree(UV, order)
    end

    # update built tree
    for UV_i in UV
        L_i = view(L, UV_i.i1:UV_i.i1+UV_i.bsz-1, UV_i.j1:UV_i.j1+UV_i.bsz-1)
        updateTree!(UV_i, L_i, tol)
    end

    return (B, UV)
end

"""
    uncompress()
    To be Declared
    generates the explicit matrix lower triangular part only
"""
function uncompress(B::Array{LowerTriangular{T,Array{T,2}},1}, UV::Array{Tnode,1}) where T<: AbstractFloat
    
    nb = length(B)
    m = size(B[1], 1)
    L = zeros(T, nb*m, nb*m)
    L = LowerTriangular(L)
    for i in 1:nb
        L[(i-1)*m+1:i*m, (i-1)*m+1:i*m] = B[i]
    end

    for UV_i in UV
        L[UV_i.i1:UV_i.i1+UV_i.bsz-1, UV_i.j1:UV_i.j1+UV_i.bsz-1] = UV_i.U * UV_i.V'
    end

    return(L)
end


"""
    displayTree()
    Display H-matrix
"""
function displayTree(Tree::Array{Tnode,1})
    for UV_i in Tree
        block = [UV_i.i1:UV_i.i1+UV_i.bsz-1, UV_i.j1:UV_i.j1+UV_i.bsz-1]
        rank = UV_i.rank
        println("Block: ", block, ", Rank: ", rank)
    end
end

"""
    hstats()
    To be Declared
"""
function hstats()
    return 0
end

"""
    permute Tree by block ordering
"""
function permuteTree(UV, order::Vector{Int})
    # uvsize = length(UV)
    # UVnew = zeros(uvsize)
end

"""
    inorder()
    compute block order
"""
function inorder()

end