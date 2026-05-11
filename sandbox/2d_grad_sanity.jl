# sandbox/2d_grad_sanity.jl
#
# Phase 0 Task 0.4 — Zygote gradient flow through a mock allgather rrule.
#
# Purpose:
#   Verify the ChainRulesCore rrule pattern we will use for production
#   `allgather_dim` works under Zygote — specifically that defining
#   `rrule(::typeof(my_allgather), x) -> (y, back)` where `back(d_y) = d_x`
#   (with d_x being a properly-shaped local slice of d_y) gives correct
#   gradients for a loss = sum(abs2, my_allgather(x)).
#
#   This is a single-rank mock (no MPI). For production:
#     forward: allgather_dim(x_local) -> x_full   (MPI Allgatherv)
#     backward: reduce_scatter_dim(d_y, ...)      (MPI Reduce_scatter / Allreduce+slice)
#   Backward of allgather IS reduce_scatter; here in 1-rank mock both are identity.
#
# When to run:
#   julia --project=. sandbox/2d_grad_sanity.jl
#
# Expected output:
#   [grad sanity] all 3 Zygote gradient checks PASS

using Zygote, ChainRulesCore, Test, LinearAlgebra

# --- Mock primitives (1-rank degenerate case) ---

"""
    my_allgather(x_local) -> x_full

In the 1-rank mock case, this is identity. In production, this is the
MPI collective that broadcasts x_local to all ranks along a sub-comm.
"""
my_allgather(x_local) = copy(x_local)

"""
    my_reduce_scatter(x_full) -> x_local

In the 1-rank mock case, this is identity. In production, this sums
x_full across all ranks in a sub-comm and keeps the rank-local slice.
"""
my_reduce_scatter(x_full) = copy(x_full)

# --- rrules: backward of allgather IS reduce_scatter, and vice versa ---

function ChainRulesCore.rrule(::typeof(my_allgather), x_local)
    y = my_allgather(x_local)
    function back(d_y)
        return NoTangent(), my_reduce_scatter(unthunk(d_y))
    end
    return y, back
end

function ChainRulesCore.rrule(::typeof(my_reduce_scatter), x_full)
    y = my_reduce_scatter(x_full)
    function back(d_y)
        return NoTangent(), my_allgather(unthunk(d_y))
    end
    return y, back
end

# --- Tests ---

println("[grad sanity] mock allgather/reduce_scatter gradient checks (1-rank)")

# Test 1: gradient of sum(abs2, allgather(x)) w.r.t. x should be 2x
let
    x = randn(Float64, 4, 8)
    g = Zygote.gradient(x -> sum(abs2, my_allgather(x)), x)[1]
    @test g ≈ 2 * x
    println("  PASS  ∂(sum(abs2, allgather(x)))/∂x ≈ 2x   (n=$(length(x)))")
end

# Test 2: gradient of sum(abs2, reduce_scatter(x)) w.r.t. x should be 2x
let
    x = randn(Float64, 8, 8)
    g = Zygote.gradient(x -> sum(abs2, my_reduce_scatter(x)), x)[1]
    @test g ≈ 2 * x
    println("  PASS  ∂(sum(abs2, reduce_scatter(x)))/∂x ≈ 2x   (n=$(length(x)))")
end

# Test 3: Composition over complex tensors.
# Note: Zygote's gradient convention for f: C^n → R is to return the
# "Wirtinger" gradient ∂f/∂(conj z), so for f(z) = Σ|z|² = Σ z·conj(z) we get
# g = 2·z (NOT 2·conj(z)). This matters for the 2D-distributed VUMPS rrules:
# our primitives just pass d_y through reduce_scatter / allgather, no manual
# conj() — Zygote's complex convention flows through unchanged.
# (This expectation was wrong in the initial draft; first Sofia run caught it.)
let
    x = randn(ComplexF64, 4, 6)
    f(x) = sum(abs2, my_reduce_scatter(my_allgather(x)))
    g = Zygote.gradient(f, x)[1]
    @test g ≈ 2 * x
    println("  PASS  ∂(sum(abs2, reduce_scatter(allgather(z))))/∂z ≈ 2z   (complex, n=$(length(x)))")
end

println("[grad sanity] all 3 Zygote gradient checks PASS")
