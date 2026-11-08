using AMDGPU
using Test

@testset "AMDGPU" begin
    times = []
    for N in 2 .^ (7:15)
        A = AMDGPU.rand(ComplexF64, N,N)
        ts = []
        for _ in 1:100
            t = AMDGPU.@elapsed A*A
            push!(ts, t*1000)
        end
        push!(times, minimum(ts))
        print("{$N, $(minimum(ts))},")
    end
end

