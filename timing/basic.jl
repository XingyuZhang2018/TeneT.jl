using TeneT: simple_eig
using VectorInterface
using BenchmarkTools
using CUDA
using LinearAlgebra
using Test
using ProfileView
using Random
using Zygote
using KrylovKit
using Printf
using OMEinsum

CUDA.allowscalar(false)
@testset "dot $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    N = 10^2
    A = atype(rand(ComplexF64, N,N))
    B = [A]
    C = [B]
    D = [C]
    # @btime CUDA.@sync inner($A, $A)
    # @btime CUDA.@sync inner($B, $B)
    # @btime CUDA.@sync inner($C, $C)
    # @btime CUDA.@sync inner($D, $D)
    function foo(x)
        for _ in 1:100
            CUDA.@sync inner(x, x)
        end
        
    end
    ProfileView.@profview foo(D)
end

@testset "dot $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    N = 10^2
    A = atype(rand(ComplexF64, N,N))
    B = [A]
    C = [B]
    D = [C]
    @btime CUDA.@sync inner($A, $A)
    @btime CUDA.@sync inner($B, $B)
    @btime CUDA.@sync inner($C, $C)
    @btime CUDA.@sync inner($D, $D)
end

@testset "eigsolve $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    N = 10^3
    A = [atype(rand(ComplexF64, N, N)) for i in 1:4]
    v0 = [atype(rand(ComplexF64, N)) for i in 1:4]
    linearmap(v) = A .* v
    @btime CUDA.@sync inner($v0, $v0)
    @btime CUDA.@sync $linearmap($v0)
    @btime CUDA.@sync λs, vs = eigsolve(v -> $linearmap(v), $v0, 1, :LM)
end
    # function foo()
    #     for _ in 1:100
    #         λs, vs = eigsolve(v -> linearmap(v), v0, 1, :LM)
    #     end
    # end
    # ProfileView.@profview foo()
@testset "eigsolve $atype N=$N" for atype in [Array, CuArray], N in [10^3, 5*10^3]
    Random.seed!(100)
    A = atype(rand(ComplexF64, N, N))
    v0 = atype(rand(ComplexF64, N))
    linearmap(v) = A * v
    @btime CUDA.@sync $A * $v0
    @btime CUDA.@sync λs, vs = eigsolve(v -> $linearmap(v), $v0, 1, :LM)
end

@testset "eigsolve $atype" for atype in [Array]
    Random.seed!(100)
    N = 10^3
    A = [atype(rand(ComplexF64, N, N)) for i in 1:1]
    v0 = [atype(rand(ComplexF64, N, N)) for i in 1:1]
    linearmap(v) = A .* v
    # @btime CUDA.@sync $A .* $v0
    # @btime CUDA.@sync λs, vs = eigsolve(v -> $linearmap(v), $v0, 1, :LM)
    # @btime CUDA.@sync $simple_eig($linearmap, $v0)
    λ, v = simple_eig(linearmap, v0)
    @show λ norm(λ*v - linearmap(v))

    # ProfileView.@profview λs, vs = eigsolve(v -> A*v, v0, 1, :LM)
end

@testset "linsolve $atype" for atype in [Array, CuArray]
    Random.seed!(100)
    N = 10^2
    A = atype(rand(ComplexF64, N, N))
    b = atype(rand(ComplexF64, N))
    # @btime CUDA.@sync $A \ $b
    # @btime CUDA.@sync linsolve(X -> $A * X, $b; maxiter = 100)
    # @btime CUDA.@sync x, stats = bilq($A, $b)
    # ProfileView.@profview linsolve(X -> A * X, b; maxiter = 100)
    # x, stats = bilq(X -> A * X, b)
end

@testset "*" begin
    D = 10^2
    A = rand(D,D)
    B = rand(D,D)
    function foo1(A, B)
        # C = similar(A)
        ein"ab,bc->ac"(A, B)
    end
    @time foo1(A, B)

    function foo2(A, B)
        C = similar(A)
        for i in 1:100
            view(C, :,i) .= ein"ab,b->a"(A, view(B, :,i))
        end
        return C
    end
    @test foo1(A, B) ≈ foo2(A, B)

    @time foo2(A, B)
end

@testset "FLmap" begin
    function meminfo_julia()
        # @printf "GC total:  %9.3f MiB\n" Base.gc_total_bytes(Base.gc_num())/2^20
        # Total bytes (above) usually underreports, thus I suggest using live bytes (below)
        @printf "GC live:   %9.3f MiB\n" Base.gc_live_bytes()/2^20
        @printf "JIT:       %9.3f MiB\n" Base.jit_total_bytes()/2^20
        @printf "Max. RSS:  %9.3f MiB\n" Sys.maxrss()/2^20
    end

    function FLmap1(FL, ALu, ALd, M)
        ein"((adf,fgh),dgeb),abc -> ceh"(FL, ALd, M, ALu)
    end

    function FLmap2(FL, ALu, ALd, M)
        FLo = similar(ALd)
        for h in 1:size(ALd, 3)
            FLo[:,:,h] = ein"((adf,fg),dgeb),abc -> ce"(FL, view(ALd, :,:,h), M, ALu)
        end

        return FLo
    end

    χ = 10
    D = 2
    FL = rand(ComplexF64, χ, D^2, χ)
    ALu = rand(ComplexF64, χ, D^2, χ)
    ALd = rand(ComplexF64, χ, D^2, χ)
    M = rand(ComplexF64, D^2, D^2, D^2, D^2)
    
    # meminfo_julia()
    # GC.gc()
    @test FLmap1(FL, ALu, ALd, M) ≈ FLmap2(FL, ALu, ALd, M)

    @time FLmap1(FL, ALu, ALd, M)
    @time FLmap2(FL, ALu, ALd, M)
end