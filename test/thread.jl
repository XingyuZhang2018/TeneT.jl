using Base.Threads
using BenchmarkTools
using CUDA
using LinearAlgebra: norm
using Random
using Test

@testset "threads" begin
    acc = Ref(0)
    for i in 1:1000
        acc[] += 1
    end
    @show acc[]
end

@testset "threads push!" begin
    solution_data = Vector{Vector{Float64}}()
    for i in 1:nthreads()
        push!(solution_data, Float64[])
    end
    for i in 1:10000
        push!(solution_data[threadid()], 1)
    end
    @show sum(vcat(solution_data...))
end

@testset "thread U1" begin
    qn_para = Vector{Vector{Vector{Int}}}()
    for i in 1:nthreads()
        push!(qn_para, Vector{Vector{Int}}())
    end
    f!(x) = push!(x, [1,1]) 
    for i in 1:10
        f!(qn_para[threadid()])
    end
    @show vcat(qn_para...)
end

@testset "CuArray of Array" begin
    Random.seed!(100)
    A = [CUDA.rand(ComplexF64, 10^3, 10^3) for i in 1:nthreads()]

    function foo_serial(A)
        A_para = Vector{CuArray{ComplexF64}}()
        for _ in 1:nthreads()
            push!(A_para, CuArray{ComplexF64}[])
        end
        for i in 1:nthreads()
            A_para[i] = A[i] * A[i]
        end
        reduce(+, A_para)
    end

    function foo_para(A)
        A_para = Vector{CuArray{ComplexF64}}()
        for _ in 1:nthreads()
            push!(A_para, CuArray{ComplexF64}[])
        end
        for i in 1:nthreads()
            A_para[threadid()] = CUDA.@sync A[i] * A[i]
        end
        reduce(+, A_para)
    end

    @test foo_serial(A) ≈ foo_para(A)
    # @btime CUDA.@sync $foo_serial($A)
    # @btime CUDA.@sync $foo_para($A)
end