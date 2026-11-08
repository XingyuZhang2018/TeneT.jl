using Test

function nccl_slice2d_section()
    src = read(joinpath(@__DIR__, "..", "src", "contraction", "parallel", "nccl_wrapper.jl"), String)
    m = match(r"# ─── Slice2D row/col allgather \+ reduce-scatter via NCCL ─+([\s\S]*?)# Equal per-rank", src)
    @assert m !== nothing
    return m.match
end

@testset "Slice2D NCCL first-leg path avoids CUDA JIT kernels" begin
    section = nccl_slice2d_section()
    @test !occursin("CUDA.@cuda", section)
    @test !occursin("_nccl_transpose2d_kernel!", section)
    @test !occursin("permutedims(", section)
    @test occursin("CUDA.CUBLAS.geam!", section)
end
