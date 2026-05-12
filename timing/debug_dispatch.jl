# Diagnose whether KrylovKit.eigsolve(::CuArray) is actually routed to GPUKrylov
# or to KrylovKit native after `delete_method`.

using TeneT, CUDA, LinearAlgebra, KrylovKit, Random, Printf

import Pkg
gpukrylov_path = "D:/1 - research/1.19 - GPU/GPUKrylov.jl"
if !haskey(Pkg.project().dependencies, "GPUKrylov")
    Pkg.develop(path=gpukrylov_path)
end
using GPUKrylov

# Type-piracy
function KrylovKit.eigsolve(f::Function, x₀::CuArray{T,N}, howmany::Int, which::Symbol;
                            tol::Real=1e-12, krylovdim::Int=30, maxiter::Int=100,
                            ishermitian::Bool=false, alg_rrule=nothing, verbosity::Int=0,
                            kwargs...) where {T, N}
    @info "→ Routed to GPUKrylov!"
    opts = GPUKrylov.ArnoldiOpts(nev=howmany, krylov_dim=krylovdim, tol=tol, maxiter=maxiter, verbosity=0)
    f! = (out, vin) -> (copyto!(out, f(vin)); out)
    λs, vs, info = N == 1 ? GPUKrylov.eigsolve!(f!, x₀, opts) :
                            GPUKrylov.eigsolve_array!(f!, x₀, opts)
    return λs, vs, (converged=info.converged, numiter=info.numiter, numops=info.numops,
                    residuals=info.residuals,
                    normres=isempty(info.residuals) ? 0.0 : info.residuals[1])
end

println("=== Methods of KrylovKit.eigsolve BEFORE delete ===")
for m in methods(KrylovKit.eigsolve)
    sig_str = string(m.sig)
    if occursin("CuArray", sig_str)
        println("  [WILL DELETE] ", m, "  sig=", sig_str)
    end
end

# Test 1: call eigsolve, see if it's routed
N = 100
A = CUDA.randn(ComplexF64, N, N) ./ sqrt(N)
v0 = CUDA.randn(ComplexF64, N); v0 ./= norm(v0)
f = v -> A * v

println("\n=== Call 1 (with type-piracy active) ===")
λ, v, info = KrylovKit.eigsolve(f, v0, 1, :LM; tol=1e-10, krylovdim=30, maxiter=10)
@printf("  λ = %s, info.numiter = %d\n", λ[1], info.numiter)

println("\n=== Now deleting type-piracy method ===")
deleted = 0
for m in methods(KrylovKit.eigsolve)
    sig_str = string(m.sig)
    if occursin("CuArray", sig_str)
        try
            Base.delete_method(m)
            deleted += 1
            println("  DELETED: ", m)
        catch e
            println("  FAILED to delete: ", m, " — ", e)
        end
    end
end
println("  Total deleted: ", deleted)

println("\n=== Methods of KrylovKit.eigsolve AFTER delete ===")
for m in methods(KrylovKit.eigsolve)
    sig_str = string(m.sig)
    if occursin("CuArray", sig_str)
        println("  [STILL HERE] ", m)
    end
end

println("\n=== Call 2 (after delete; should be native KrylovKit) ===")
try
    λ, v, info = KrylovKit.eigsolve(f, v0, 1, :LM; tol=1e-10, krylovdim=30, maxiter=10)
    @printf("  λ = %s, info.numiter = %d\n", λ[1], info.numiter)
catch e
    println("  ERROR: ", e)
end

# Test 2: time both paths separately
println("\n=== Timing check ===")

# Re-add the method
@eval function KrylovKit.eigsolve(f::Function, x₀::CuArray{T,N}, howmany::Int, which::Symbol;
                                   tol::Real=1e-12, krylovdim::Int=30, maxiter::Int=100,
                                   ishermitian::Bool=false, alg_rrule=nothing, verbosity::Int=0,
                                   kwargs...) where {T, N}
    opts = GPUKrylov.ArnoldiOpts(nev=howmany, krylov_dim=krylovdim, tol=tol, maxiter=maxiter, verbosity=0)
    f! = (out, vin) -> (copyto!(out, f(vin)); out)
    λs, vs, info = N == 1 ? GPUKrylov.eigsolve!(f!, x₀, opts) :
                            GPUKrylov.eigsolve_array!(f!, x₀, opts)
    return λs, vs, (converged=info.converged, numiter=info.numiter, numops=info.numops,
                    residuals=info.residuals,
                    normres=isempty(info.residuals) ? 0.0 : info.residuals[1])
end

# Warmup
KrylovKit.eigsolve(f, v0, 1, :LM; tol=1e-10, krylovdim=30, maxiter=10)
CUDA.synchronize()

# With piracy
t_with = @elapsed begin
    for _ in 1:10
        KrylovKit.eigsolve(f, v0, 1, :LM; tol=1e-10, krylovdim=30, maxiter=10)
    end
    CUDA.synchronize()
end
@printf("With type-piracy (GPUKrylov): %.3f ms/call\n", t_with*1000/10)

# Delete piracy and time native
for m in methods(KrylovKit.eigsolve)
    if occursin("CuArray", string(m.sig))
        Base.delete_method(m)
    end
end
KrylovKit.eigsolve(f, v0, 1, :LM; tol=1e-10, krylovdim=30, maxiter=10)
CUDA.synchronize()

t_native = @elapsed begin
    for _ in 1:10
        KrylovKit.eigsolve(f, v0, 1, :LM; tol=1e-10, krylovdim=30, maxiter=10)
    end
    CUDA.synchronize()
end
@printf("Native KrylovKit:              %.3f ms/call\n", t_native*1000/10)
@printf("Ratio: %.2fx\n", t_native/t_with)
