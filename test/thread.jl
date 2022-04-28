using Test

@testset "threads" begin
    acc = Ref(0)
    Threads.@threads for i in 1:1000
        acc[] += 1
    end
    @show acc[]
end

@testset "threads push!" begin
    solution_data = Vector{Vector{Float64}}()
    for i in 1:Threads.nthreads()
        push!(solution_data, Float64[])
    end
    Threads.@threads for i in 1:10000
        push!(solution_data[Threads.threadid()], 1)
    end
    @show sum(vcat(solution_data...))
end

@testset "thread U1" begin
    qn_para = Vector{Vector{Vector{Int}}}()
    for i in 1:Threads.nthreads()
        push!(qn_para, Vector{Vector{Int}}())
    end
    f!(x) = push!(x, [1,1]) 
    Threads.@threads for i in 1:10
        f!(qn_para[Threads.threadid()])
    end
    @show vcat(qn_para...)
end