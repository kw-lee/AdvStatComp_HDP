using LinearAlgebra
using Distributions
using LowRankApprox

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
    opts = LRAOptions(maxdet_tol=0., sketch_randn_niter=1)
    opts.sketch = :randn
    opts.rtol = 5*eps(real(Float64))
    
    A = Matrix(A)
    n = size(A)[1]
    nlev = Int(floor(log2(n/m)))
    for i in 1:nlev
        nb = n ÷ 2^i
        xbegin = 0; ybegin = nb
        
        # kmax denote max rank. better heuristic needed here
<<<<<<< HEAD
        kmax = 16 + Int(floor(sqrt(nb)))
        kmax = kmax < nb ÷ 2 ? kmax : nb ÷ 2
        for j in 1:2^(i - 1)
            U, S, V = psvd(A[(xbegin + 1):(xbegin + nb), (ybegin + 1):(ybegin + nb)], opts, rank = kmax)
            if(size(U)[2] < kmax)
                kmax = size(U)[2]
            end
=======
        kmax = 8 + Int(floor(sqrt(nb)))
        kmax = kmax < nb ÷ 2 ? kmax : nb ÷ 2
        for j in 1:2^(i - 1)
            U, S, V = psvd(A[(xbegin + 1):(xbegin + nb), (ybegin + 1):(ybegin + nb)], opts, rank = kmax)
>>>>>>> cmvn
            
            A[(xbegin + 1):(xbegin + nb), (ybegin + 1):(ybegin + kmax)] = U * diagm(sqrt.(S))
            A[(xbegin + 1):(xbegin + nb), (ybegin + kmax + 1):(ybegin + nb)] .= 0
            A[(ybegin + 1):(ybegin + nb), (xbegin + 1):(xbegin + kmax)] = V * diagm(sqrt.(S))
            A[(ybegin + 1):(ybegin + nb), (xbegin + kmax + 1):(xbegin + nb)] .= 0
            
            xbegin += nb * 2; ybegin += nb * 2
        end
    end
    
    nb = n ÷ 2^nlev
    xbegin = 0; ybegin = 0
    for j in 1:2^nlev
        L = A[(xbegin + 1):(xbegin + nb), (ybegin + 1):(ybegin + nb)]
        A[(xbegin + 1):(xbegin + nb), (ybegin + 1):(ybegin + nb)] = Matrix(cholesky(L).L)
        xbegin += nb; ybegin += nb
        end
    
    return A
end

function uncompress(A::Array{T,2}, m::Int) where T<: AbstractFloat
    n = size(A)[1]
    if(n == m)
        return A = A * transpose(A)
    end
    nb = n ÷ 2
    A[1:nb, 1:nb] = uncompress(A[1:nb, 1:nb], m)
    A[(nb + 1):n, 1:nb] = A[(nb + 1):n, 1:nb] * transpose(A[1:nb, (nb + 1):n])
    A[1:nb, (nb + 1):n] = transpose(A[(nb + 1):n, 1:nb])
    A[(nb + 1):n, (nb + 1):n] = uncompress(A[(nb + 1):n, (nb + 1):n], m)
    return A
end