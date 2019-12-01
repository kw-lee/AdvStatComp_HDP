include("hchol.jl")
include("mvn.jl")

"""
    hmvn(B, UV, a1, b1, ns, N; tol = 1e-8, mu = 0)
    input: 
        - B, UV: h_cholesky factor of the covariance matrix
        - a1: lower bound
        - b1: upper bound
        - ns: The number of sample size
        - N: Randomized QMC points
        - tol: tolerance
        - mu: mean
    output:
        - p_mean: estimated probabiliy
"""
function hmvn(B::Array{LowerTriangular{T,Array{T,2}},1}, 
    UV::Array{Tnode,1}, a1::Vector{T}, b1::Vector{T}, 
    ns::Int, N::Int; tol = convert(T, 1e-8),
    μ::Array{T,1} = zeros(T, length(a1))) where T<:AbstractFloat

    a1 -= μ # centering
    b1 -= μ # centering

    nb = length(B) # the number of blocks
    m = size(B[1], 1) # block size
    n = nb * m # total size of

    # values produced by the ns samples, each with N randomized qmc points
    values = Vector{T}(undef, ns) 

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

    vp = Matrix{T}(undef, nb, N)
    y = Matrix{T}(undef, nb*m, N)

    for i in 1:ns
        xr = rand(T, m, 1) # xr ~ U(0,1)
        a = reshape(repeat(a1, N), n, N)
        b = reshape(repeat(b1, N), n, N)

        for r in 1:nb
            r1 = (r-1)*m
            if r > 1
                i1 = UV[r-1].i1
                j1 = UV[r-1].j1
                bsz = UV[r-1].bsz
                delta = zeros(T, bsz, N) # g in Cao2019
                if UV[r-1].rank != 0
                    delta .= UV[r-1].U * ( transpose(UV[r-1].V) * y[j1+1:j1+bsz,1:N] )
                end
                a[i1:i1+bsz-1, 1:N] -= delta 
                b[i1:i1+bsz-1, 1:N] -= delta 
            end
            X = Matrix{T}(undef, m, N)
            for j in 1:N
                X[:,j] = view(q, r1+1:r1+m) * (1+j) + xr
            end
            X = map(x->abs(2*(x-floor(x))-1), X)
            pr, yr = mvndns(m, N, B[r], X, a[r1+1:r1+m,1:N], b[r1+1:r1+m, 1:N], tol)

            vp[r, :] .= pr
            # yr is an monte-carlo approximator of the solve(B, expt{x})
            # expt{x} can be calcultaed by 
            y[r1+1:r1+m, 1:N] .= yr 
        end

        p = Vector{T}(undef, N)
        for j in 1:N
            p[j] = prod(filter(x -> !isnan(x), vp[1:nb, j])) # omit nan values
        end
        values[i] = mean(p)
    end

    p_mean = mean(values) # estimated probabiliy
    return p_mean
end