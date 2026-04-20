# Wengert.jl CPU-offload checkpoint for QRCTM — Design

Date: 2026-04-20
Scope: Replace per-step recompute `checkpoint` in QRCTM AD loop with
[Wengert.jl](https://github.com/XingyuZhang2018/Wengert.jl) `@checkpoint`
(CPU-offload mode) + `barrier` bridge to Zygote. First target is
`examples/Heisenberg/Heisenberg_Square_QRCTM.jl`, D=7 χ=256 on RTX 4090.

## Motivation

Current QRCTM uses `checkpoint(qrctm_step, env, M, alg)` at
[`src/boundary_algorithm/qrctm.jl:120`](src/boundary_algorithm/qrctm.jl). It is
a Zygote.@adjoint that throws forward intermediates away and re-runs
`Zygote._pullback` on backward. Peak device memory per step ≈ one step's
FLmap intermediates; between steps the only surviving state is `env`.

Wengert.jl offers a *true* CPU-offload checkpoint: intermediate **tape slots**
are moved to host RAM after forward and reloaded on backward. The invariant
required to realise the saving is:

- the memory-heavy activations must live on a **Wengert tape slot**, not
  inside a Zygote closure (`zpb`).

That constrains the design shape: outer AD ownership must belong to Wengert
for the checkpointed region, and each Zygote sub-call must sit behind a
`barrier(…, Zygote.pullback, …)` whose output becomes a new slot on the
Wengert tape.

## Architecture (accepted)

```
Zygote energy fn        (outer optimiser AD — unchanged)
 └─ Zygote.@adjoint checkpoint_wengert_loop(env, M, alg, …)
     └─ Wengert.pullback(env, M) do e, m          # outer = Wengert tape
          cur = e
          for i in 1:alg.maxiter_ad
            alg_this = choose(alg, i)              # mixed-precision polish
            cur = @checkpoint                      # mark slots for CPU offload
                  qrctm_step_split(cur, m, alg_this)
          end
          return cur                               # single tracked output
     end
     # back = wback, zpb's inside barriers released as Wengert walks backward
```

where `qrctm_step_split` mirrors `qrctm_step` but splits its body into
`barrier` calls:

```julia
function qrctm_step_split(env, M, alg)
    C = env.C; T = env.T

    CT    = barrier((c,t) -> _to_front(CTtoT(c,t)),     Zygote.pullback, C, T)
    U, R  = barrier(qr_for_ad,                           Zygote.pullback, CT)
    U     = barrier(u -> reshape(u, size(T)),            Zygote.pullback, U)
    T_new = barrier(
        (t,u1,u2,m) -> FLmap_parallel(t,u1,u2,m;
             ifparallel=alg.ifparallel, forloop_iter=alg.forloop_iter,
             inner_etype=alg.inner_etype),
        Zygote.pullback, T, U, U, M)
    C_new = barrier(Cmap,                                Zygote.pullback,
                    R, T_new, U)

    T_new /= Wengert.@ignore norm(T_new)
    C_new /= Wengert.@ignore norm(C_new)
    err    = Wengert.@ignore norm(C_new - C)

    return CTMEnv(C_new, T_new), err
end
```

Key properties:

1. **One Wengert tape spans the whole AD loop** → inter-step `env` slots are
   offloadable to CPU under `@checkpoint`.
2. **Within a step, `barrier` boundaries break up heavy activations** — the
   output of `FLmap_parallel`, `CTtoT`, `qr`, `Cmap` each become their own
   tape slot. Slots offload individually.
3. **Zygote owns each small chunk's rrule** — every TensorOperations-heavy
   sub-op stays inside a `Zygote.pullback` captured by `barrier`, which is
   what currently makes qrctm differentiable at all.
4. **zpb closures stay device-local for a single small op only** — their
   captured intermediates are small (per-sub-op, not whole-step).

### Why not simpler shapes?

- **Per-step Zygote.@adjoint wrapping one Wengert.pullback**: each step gets
  its own tape. Tape is captured in Zygote's chain → all maxiter_ad tapes
  live simultaneously during Zygote backward → zpb closures pile up on
  device. No offload benefit.
- **Loop-level Wengert + single big barrier per step**: inter-step env slots
  are offloaded, but the big `zpb` holding FLmap intermediates is still in
  one TapeEntry per step → maxiter_ad of them live on device → still OOMs.
- **Full Wengert replacement**: blocked — Wengert currently can't trace
  through TensorOperations (no ChainRules rrules reachable on those paths).
  `barrier` is the prescribed escape hatch.

## Changes by file

### `Project.toml`

```toml
[deps]
# …existing…
Wengert = "42a33ce9-1e22-44b5-bec4-f090074a6e82"

[sources]
Wengert = {url = "https://github.com/XingyuZhang2018/Wengert.jl"}
```

Fallback for Julia <1.11: `Pkg.add(url="https://github.com/XingyuZhang2018/Wengert.jl")`
— documented in README / commit message.

### `src/TeneT.jl`

Add `using Wengert` (near other AD imports).

### `src/boundary_algorithm/environment.jl`

Add one line at CTMEnv definition site:

```julia
Functors.@functor CTMEnv  # enables Wengert to wrap C and T as TrackedArrays
```

### `src/boundary_algorithm/interface.jl`

Add to `QRCTM` kwstruct next to `ifcheckpoint`:

```julia
ifcheckpoint_wengert::Bool = false
```

### `src/boundary_algorithm/qrctm.jl`

- Add `qrctm_step_split(env, M, alg)` (body above) — only used when Wengert
  path is active.
- Replace the AD-phase loop body's `alg.ifcheckpoint ? checkpoint(…) : …`
  line with a **loop-level** 3-way dispatch:

```julia
if alg.ifcheckpoint_wengert
    env, err = checkpoint_wengert_loop(env, M, alg, alg_ad, alg_ad_coarse,
                                       mixed_active, want_whole, T_orig)
    break  # the loop is consumed inside the adjoint
elseif alg.ifcheckpoint
    # …existing per-step branch…
else
    # …existing…
end
```

Actual shape: factor today's `for i in 1:alg.maxiter_ad` into an inner
function so `checkpoint_wengert_loop` can call the same logic (including
polish-iter cast-back, verbosity prints via `ignore_derivatives`, early
`break` on `err < tol`). The Wengert branch runs the whole loop inside the
adjoint; the other two branches keep today's step-at-a-time structure.

### `src/utils/misc.jl`

Add (near existing `checkpoint`/`checkpoint_offload`):

```julia
checkpoint_wengert_loop(loop_body_fn, env, M, alg, extras...) =
    loop_body_fn(env, M, alg, extras...)

Zygote.@adjoint function checkpoint_wengert_loop(loop_body_fn, env, M, alg, extras...)
    local err_captured
    env_final, wback = Wengert.pullback(env, M) do e, m
        env_r, err_r = @checkpoint loop_body_fn(e, m, alg, extras...)
        err_captured = err_r
        env_r
    end
    return (env_final, err_captured), Δ -> begin
        Δenv, _ = Δ
        genv, gM = wback(Δenv)
        (nothing, genv, gM, nothing, ntuple(_ -> nothing, length(extras))...)
    end
end
```

Guard: if `alg.ifcheckpoint && alg.ifcheckpoint_wengert` throw
`ArgumentError("ifcheckpoint and ifcheckpoint_wengert are mutually exclusive")`
at `leading_boundary` entry.

## Test / verification

1. **Gradient equivalence** (small case): run
   `Heisenberg_Square_QRCTM.jl` with D=2 χ=16 maxiter_ad=3 under three
   settings: `ifcheckpoint=false`, `ifcheckpoint=true`,
   `ifcheckpoint_wengert=true`. Compare the gradient of one fg call. All
   three should agree to ≈1e-10 (Float64 ULP).
2. **Local 4090 target**: D=7 χ=256, maxiter_ad=20. Record
   - peak `CUDA.memory_info()` during fg
   - wall time per fg
   - first optimiser step gradient norm
   vs `ifcheckpoint=true` baseline on the same seed.

## Out of scope

- Applying Wengert checkpoint to VUMPS sub-maps (FLmap/ACmap/Cmap/vumps_step)
  — follow-up if QRCTM goes well.
- Replacing `Zygote.pullback` for the outer energy gradient.
- Retiring the existing `checkpoint` / `checkpoint_offload` paths.

## Risks

- **Wengert mixed-precision interaction**: `inner_etype`/`whole_vumps_etype`
  change eltype of env mid-loop. Wengert tape slots store the current value;
  an on-the-fly `_downcast_eltype` inside the pullback body may need to go
  through a `barrier` itself. Will be exercised by the small-D verification
  step.
- **`barrier` + kwargs**: `FLmap_parallel` takes keyword args. The Wengert
  `barrier(f, ad_pullback, args…)` API is positional; the design uses an
  anonymous wrapper `(t,u1,u2,m) -> FLmap_parallel(t,u1,u2,m; kwargs…)` to
  bake in kwargs — verified against `_make_barrier_pb` pattern in
  `Wengert/src/api.jl`.
- **`Functors.@functor CTMEnv`**: touches a shared type used everywhere. If
  it changes behaviour in an unrelated codepath (Zygote's own functoring of
  env, e.g. in saving/loading), revert to pass-`(C,T)`-tuples design.
