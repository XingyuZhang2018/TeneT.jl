# Diagnostic: verify F32 actually flows through the env solvers.
# If env-level cast works, eltype should be ComplexF32 at every probe point.
# If something silently reverts to F64, we'll see it here.

using Random, CUDA, MPI, TeneT, LinearAlgebra, Zygote, Printf
const StructArray = TeneT.StructArray

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
nprocs = MPI.Comm_size(comm)
CUDA.device!(0)

seed = 42; Random.seed!(seed)
D = 10; chi = 400; forloop_iter = 32

rank == 0 && println("=== eltype-leak diagnostic  nprocs=$nprocs D=$D chi=$chi ===")

# Build small-ish env matching Plaquette setup
pattern = [1 3; 2 4]
model = J1J2(lattice=Square(), S=0.5, J1=1.0, J2=0.5, ifrotate=true, couplingtype=:uniform, bondratio=1.0)

function restriction_ipeps(A)
    Ar = Zygote.Buffer(A)
    Ar[:,:,:,:,:,1] = A[:,:,:,:,:,1]
    Ar[:,:,:,:,:,1] += permutedims(Ar[:,:,:,:,:,1], (4,3,2,1,5))
    Ar[:,:,:,:,:,2] = permutedims(Ar[:,:,:,:,:,1], (1,4,3,2,5))
    Ar[:,:,:,:,:,3] = permutedims(Ar[:,:,:,:,:,1], (3,2,1,4,5))
    Ar[:,:,:,:,:,4] = permutedims(Ar[:,:,:,:,:,1], (3,4,1,2,5))
    return copy(Ar)
end

alg_f32 = VUMPS{TeneT.Plaquette{Square}}(
    ifsimple_eig=true, ifparallel=true, ifcheckpoint=true,
    forloop_iter=forloop_iter,
    maxiter=1, miniter=0, maxiter_ad=0, miniter_ad=0,
    power_iter=1, power_iter_ad=5, power_iter_obs=40,
    show_every=10, tol=1e-10, verbosity=0,
    inner_etype=Float32,
    inner_etype_final_steps=0,
    simple_eig_polish_steps=0,
    whole_vumps_etype=nothing,
    ifoffload_eig=false,
    ifoffload_step=false,
)

params = GradientOptimize(
    model=model, pattern=pattern, boundary_alg=alg_f32,
    optimizer=Zygote.gradient,
    ifcheckpoint=false, forloop_iter=forloop_iter, maxiter_restart=1,
    verbosity=0, folder="./_mb_tmp/", ifSU=false, SUτ=0,
    ifprecondition=false, iter_precond=0,
    reuse_env=false, ifsave_env=false, ifload_env=false,
    ifsave_lbfgs=false, ifload_lbfgs=false, ifplot=false,
)

A = TeneT.init_ipeps(; atype=CuArray, etype=Float64, No=0, D, χ=chi, params)
Ar = restriction_ipeps(A)
M_struct = TeneT.build_A(Ar, params)
rt = TeneT.init_env(M_struct, chi, alg_f32)

show_eltype(name, x) = rank == 0 && println("  $name eltype = $(x isa StructArray ? eltype(x.data[1]) : eltype(x))")

if rank == 0
    println("\n── BEFORE any env call ──")
    show_eltype("rt.AL", rt.AL)
    show_eltype("rt.C", rt.C)
    show_eltype("rt.FL", rt.FL)
    show_eltype("M", M_struct)
end

# Manually simulate what vumps_step does, with eltype probes

# --- leftenv (should cast at entry, produce F32 FL' internally) ---
rank == 0 && println("\n── Calling leftenv ──")
MPI.Barrier(comm)
AL = rt.AL
ALd = conj(AL)

# Patch leftenv to print eltypes
# Alternative: call leftenv and check returned FL eltype
λL_out, FL_out = TeneT.leftenv(AL, ALd, M_struct, rt.FL; alg=alg_f32)

if rank == 0
    println("After leftenv returns:")
    show_eltype("λL_out", λL_out)
    show_eltype("FL_out", FL_out)
    println("(expected: F64 outputs since cast-back at env exit)")
end

# --- Now ACenv_plaq ---
rank == 0 && println("\n── Calling ACenv_plaq ──")
MPI.Barrier(comm)
AC = TeneT.ALCtoAC(rt.AL, rt.C)
if rank == 0
    show_eltype("AC (before ACenv_plaq)", AC)
    show_eltype("FL_out (input to ACenv_plaq)", FL_out)
end
λAC_out, AC_out = TeneT.ACenv_plaq(AC, FL_out, M_struct; alg=alg_f32)
if rank == 0
    show_eltype("AC_out", AC_out)
end

# --- Now manually call ACmap_parallel in F32 mode explicitly ---
# To compare: directly call ACmap_parallel (which goes through parallel() with inner_etype)
# vs call ACmap_parallel with F32 pre-cast inputs and inner_etype=nothing
rank == 0 && println("\n── Probing ACmap_parallel behavior ──")
AC_elem = AC[1,1]   # a single CuArray (the AC of the top-left plaquette cell)
FL_elem1 = FL_out[1,1]
FL_elem2 = FL_out[1,2]
M_elem = M_struct[1,1]

if rank == 0
    println("Inputs to ACmap_parallel:")
    show_eltype("AC_elem", AC_elem)
    show_eltype("FL_elem1", FL_elem1)
    show_eltype("FL_elem2", FL_elem2)
    show_eltype("M_elem", M_elem)
end

# Warmup first — build cuTENSOR plans for each config.
# IMPORTANT: production tensors here are REAL Float64. Cast must preserve
# realness (Float32, not ComplexF32) to match what _boundary_cast does in
# the real env-level code path. Casting real→complex artificially doubles
# flops and gives a meaningless slowdown.
@assert !(eltype(AC_elem) <: Complex) "expected real tensors; complex would change interpretation"
AC_f32 = Float32.(AC_elem)
FL_f32_1 = Float32.(FL_elem1)
FL_f32_2 = Float32.(FL_elem2)
M_f32 = Float32.(M_elem)
rank == 0 && println("\n  Cast types: AC_f32=$(eltype(AC_f32)) (should be Float32, not ComplexF32)")

rank == 0 && println("\n  Warming up cuTENSOR plans for all 3 paths (5 iters each)...")
for _ in 1:5
    TeneT.ACmap_parallel(AC_elem, FL_elem1, FL_elem2, M_elem; ifparallel=true, forloop_iter, inner_etype=Float32); CUDA.synchronize()
    TeneT.ACmap_parallel(AC_f32, FL_f32_1, FL_f32_2, M_f32; ifparallel=true, forloop_iter, inner_etype=nothing); CUDA.synchronize()
    TeneT.ACmap_parallel(AC_elem, FL_elem1, FL_elem2, M_elem; ifparallel=true, forloop_iter, inner_etype=nothing); CUDA.synchronize()
end
MPI.Barrier(comm)

# Helper: median of N timed calls
function median_time(f, N)
    ts = Float64[]
    for _ in 1:N
        CUDA.synchronize(); MPI.Barrier(comm)
        t0 = time_ns()
        f()
        CUDA.synchronize(); MPI.Barrier(comm)
        push!(ts, (time_ns() - t0) / 1e6)
    end
    return length(ts) > 2 ? sort(ts)[end÷2+1] : ts[1]
end

t1 = median_time(() -> TeneT.ACmap_parallel(AC_elem, FL_elem1, FL_elem2, M_elem; ifparallel=true, forloop_iter, inner_etype=Float32), 10)
t2 = median_time(() -> TeneT.ACmap_parallel(AC_f32, FL_f32_1, FL_f32_2, M_f32; ifparallel=true, forloop_iter, inner_etype=nothing), 10)
t3 = median_time(() -> TeneT.ACmap_parallel(AC_elem, FL_elem1, FL_elem2, M_elem; ifparallel=true, forloop_iter, inner_etype=nothing), 10)

# r1/r2/r3 for eltype print
r1 = TeneT.ACmap_parallel(AC_elem, FL_elem1, FL_elem2, M_elem; ifparallel=true, forloop_iter, inner_etype=Float32)
r2 = TeneT.ACmap_parallel(AC_f32, FL_f32_1, FL_f32_2, M_f32; ifparallel=true, forloop_iter, inner_etype=nothing)
r3 = TeneT.ACmap_parallel(AC_elem, FL_elem1, FL_elem2, M_elem; ifparallel=true, forloop_iter, inner_etype=nothing)
CUDA.synchronize(); MPI.Barrier(comm)

if rank == 0
    println()
    @printf("Test 1 (F64 inputs + inner_etype=Float32, parallel-level cast): %7.2f ms, eltype(result)=%s\n",
            t1, eltype(r1))
    @printf("Test 2 (F32 inputs + inner_etype=nothing, env-level style):     %7.2f ms, eltype(result)=%s\n",
            t2, eltype(r2))
    @printf("Test 3 (F64 inputs + inner_etype=nothing, pure F64):            %7.2f ms, eltype(result)=%s\n",
            t3, eltype(r3))
    println()
    @printf("Relative to Test 3 (F64 baseline):\n")
    @printf("  Test 1 ratio: %.3f\n", t1/t3)
    @printf("  Test 2 ratio: %.3f (this is env-level-cast equivalent)\n", t2/t3)
    println()
    println("If Test 2 < Test 3 significantly → env-level cast + F32 kernel works.")
    println("If Test 2 ≈ Test 3 → something in the F32 path reverts to F64.")
end

MPI.Finalize()
