cd("C:/Users/spatial stat/Desktop/2019-2/고급통계계산/Project/github/AdvStatComp_HDP/src")
include("mvn.jl")
include("cmvn.jl")
include("rcmvn.jl")

iters = 250
ms = [16, 32, 64, 128]; ds = [1, 2, 4, 8, 16]
ans_1 = zeros(length(ms), length(ds), iters); time_1 = zeros(length(ms), length(ds))
ans_2 = zeros(length(ms), length(ds), iters); time_2 = zeros(length(ms), length(ds))
real = zeros(length(ms), length(ds), iters)

for i in 1:length(ms)
    for j in 1:length(ds)
        m = ms[i]; d = ds[j]
        for p in 1:iters
            Σ = Σ_generate(m)
            a = fill(-Inf, m)
            b = rand(Uniform(0, m), m)
            time_1[i, j] += @elapsed ans_1[i, j, p] += CMVN(Σ, a, b, d)[1]
            time_2[i, j] += @elapsed ans_2[i, j, p] += RCMVN(Σ, a, b, d)[1]
            real[i, j, p] += mvn(cholesky(Σ).L, a, b)
        end
    end
end
time_1 ./= iters
time_2 ./= iters