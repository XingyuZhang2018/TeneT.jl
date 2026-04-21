# Unified Checkpoint API — Design

**Date**: 2026-04-21
**Branch target**: `iPEPS-unified`
**Scope**: `src/utils/misc.jl` + `src/utils/checkpoint.jl` (new) + `src/boundary_algorithm/interface.jl` + call sites in `src/boundary_algorithm/vumps/*.jl` + 3 example files + 1 test file

## Motivation

`TeneT` currently has two parallel checkpoint functions:

- `checkpoint(f, args...)` — standard recompute (`misc.jl:79`)
- `checkpoint_offload(f, args...)` — recompute with args copied to host first (`misc.jl:132`)

and three Bool flags gating them in the `VUMPS` algorithm struct:

- `ifcheckpoint` — controls both the inner-map granularity (FLmap/FRmap/ACmap/Cmap calls inside power iter) AND the step granularity (whole `vumps_step`)
- `ifoffload_eig` — controls offload wrapper on per-row `simple_eig`
- `ifoffload_step` — controls offload wrapper on whole `vumps_step`

This triple is confusing: `ifcheckpoint` overloads two granularities with the same switch; the three Bools form an implicit 3-way dispatch at call sites; adding a future no-recompute method (e.g. Wengert-tape offload) would require yet another Bool plus more branches.

## Goal

Replace the two functions and three Bools with one dispatch-based API:

- One `CheckpointMethod` type hierarchy (`Plain` / `Recompute` / `Offload`, extensible)
- One `checkpoint(method, f, args...)` entry point using Julia multiple dispatch
- Three per-granularity fields in `VUMPS`: `inner_checkpoint`, `eig_checkpoint`, `step_checkpoint` — each independently selects a `CheckpointMethod`
- Symbol sugar via `Base.convert` so `VUMPS(...; step_checkpoint=:offload, ...)` works

Out of scope: `QRCTM.ifcheckpoint` (kept as Bool), `iPEPSOptimize.ifcheckpoint` (separate energy-loop semantic), inner-map `Offload` support (runtime-asserted off).

## Design

### Type hierarchy

In new file `src/utils/checkpoint.jl`:

```julia
abstract type CheckpointMethod end
struct Plain     <: CheckpointMethod end
struct Recompute <: CheckpointMethod end
struct Offload   <: CheckpointMethod end
```

Exported from `src/TeneT.jl`.

### Symbol coercion

```julia
_ckpt_method(m::CheckpointMethod) = m
_ckpt_method(s::Symbol) = s === :plain || s === :none ? Plain()     :
                          s === :recompute            ? Recompute() :
                          s === :offload              ? Offload()   :
                          throw(ArgumentError("unknown checkpoint method: $s"))

Base.convert(::Type{CheckpointMethod}, s::Symbol) = _ckpt_method(s)
```

One `convert` method enables Symbol to flow into `::CheckpointMethod` struct fields, e.g. via `@kwdef` constructor kwargs.

### Unified function API

```julia
# Symbol entry → normalize to singleton
checkpoint(m::Symbol, f, args...; kwargs...) = checkpoint(_ckpt_method(m), f, args...; kwargs...)

# Plain: identity, no adjoint override
checkpoint(::Plain, f, args...; kwargs...) = f(args...; kwargs...)

# Recompute: identity forward, re-run on backward
checkpoint(::Recompute, f, args...; kwargs...) = f(args...; kwargs...)
Zygote.@adjoint checkpoint(m::Recompute, f, args...; kwargs...) =
    f(args...; kwargs...),
    ȳ -> begin
        inner = Zygote._pullback((aa...) -> f(aa...; kwargs...), args...)[2](ȳ)
        (nothing, inner...)   # prepend ∂m = nothing
    end

# Offload: identity forward, offload args to host, reconstruct on backward
checkpoint(::Offload, f, args...; kwargs...) = f(args...; kwargs...)
Zygote.@adjoint function checkpoint(m::Offload, f, args...; kwargs...)
    y = f(args...; kwargs...)
    atype = _detect_target_atype(args)
    args_cpu = map(_offload_to_host, args)
    return y, function(ȳ)
        args_dev = map(a -> _to_atype(atype, a), args_cpu)
        inner = Zygote._pullback((aa...) -> f(aa...; kwargs...), args_dev...)[2](ȳ)
        (nothing, inner...)
    end
end
```

Reuses existing `_detect_target_atype`, `_offload_to_host`, `_to_atype` helpers in `misc.jl:96-130` and `environment.jl:128-159` — no changes to offload infrastructure.

Old `checkpoint(f, args...)` and `checkpoint_offload(f, args...)` signatures are **deleted**. All callers migrate.

### VUMPS struct refactor

`src/boundary_algorithm/interface.jl:11-53`:

- **Delete**: `ifcheckpoint`, `ifoffload_eig`, `ifoffload_step`
- **Add** (mutable `@kwdef`):

```julia
inner_checkpoint::CheckpointMethod = Plain()
eig_checkpoint::CheckpointMethod   = Plain()
step_checkpoint::CheckpointMethod  = Plain()
```

Symbol kwargs go through `Base.convert` automatically. No custom constructor needed.

QRCTM and iPEPSOptimize structs are unchanged.

### Call-site changes

**Step granularity** (3 sites, fully isomorphic): `c4v.jl:166-167`, `plaquette.jl:242-243`, `general.jl:964-965`:

```julia
# Old
rt, err = alg.ifoffload_step ? checkpoint_offload(vumps_step, rt, M, alg_this_iter) :
          alg.ifcheckpoint    ? checkpoint(vumps_step, rt, M, alg_this_iter) :
                                vumps_step(rt, M, alg_this_iter)

# New
rt, err = checkpoint(alg.step_checkpoint, vumps_step, rt, M, alg_this_iter)
```

**Eig granularity** (3 sites in `general.jl`: `:284, :371, :576`):

```julia
if alg.ifsimple_eig
    if alg.eig_checkpoint isa Plain && !polish_fine
        λLs, FLi1s = simple_eig(f, FL[i, 1]; power_iter)
    elseif alg.eig_checkpoint isa Plain && polish_fine
        λLs, FLi1s = simple_eig(f, FL[i, 1]; power_iter, f_final=f_polish,
                                 final_polish_steps=simple_eig_polish_steps)
    else
        λLs, FLi1s = checkpoint(alg.eig_checkpoint, _simple_eig_FLmap,
                                FL[i, 1], ALu[i, :], ALd[ir, :], M[i, :];
                                power_iter, ifparallel, forloop_iter,
                                inner_etype=inner_etype_pass,
                                final_polish_steps = polish_fine ? simple_eig_polish_steps : 0)
    end
else
    ...  # eigsolve path unchanged
end
```

Rationale: `Plain()` takes the direct `simple_eig(f, ...)` path (cheapest; `f` is a closure, segment checkpoints inside `simple_eig` do their own fine-grained work). `Recompute()` and `Offload()` both need the explicit-args `_simple_eig_FLmap` wrapper so the outer adjoint can capture / offload the neighborhood tensors cleanly. The non-redundancy of outer `Recompute()` over inner segment checkpoints: the final `v1 = f(v)` iteration in `simple_eig` (`misc.jl:44, 67`) plus `dot`/`norm`/`orth_for_ad` are NOT inside segment checkpoints — their forward activations reach the outer tape, and `Recompute()` at eig level collapses those.

FR and AC sites are isomorphic.

**Inner-map granularity** (7 sites across `c4v.jl`, `plaquette.jl`, `general.jl`):

```julia
# Old
f(FLij) = ifcheckpoint ? checkpoint(FLmap, 1, FLij, ...) : FLmap(1, FLij, ...)

# New
f(FLij) = checkpoint(alg.inner_checkpoint, FLmap, 1, FLij, ...)
```

Plus `_assert_inner_method(alg.inner_checkpoint)` helper called once at the entry of `leftenv`, `rightenv`, `ACenv`, `Cenv`:

```julia
_assert_inner_method(::Union{Plain,Recompute}) = nothing
_assert_inner_method(::Offload) = throw(ArgumentError(
    "inner-map checkpoint only supports Plain/Recompute; set eig_checkpoint=Offload() for eig-level offload instead"))
```

**Internal `_power_iter_segment` call** (`misc.jl:35, 53`): migrate old signature

```julia
v = checkpoint(_power_iter_segment, f, v, seg)
# →
v = checkpoint(Recompute(), _power_iter_segment, f, v, seg)
```

Hard-coded — not user-controlled.

### Examples migration

- 11× `ifcheckpoint=false` → delete line (Plain is default)
- 3× `ifcheckpoint=true` in VUMPS contexts (`Heisenberg_Square_VUMPS_C4v.jl:22`, `Heisenberg_Square_VUMPS_C4v_f32_smoke.jl:22`, `MPI_parallel.jl:66`) → replace with:
  ```julia
  inner_checkpoint = Recompute(),
  step_checkpoint  = Recompute(),
  ```
- QRCTM and iPEPSOptimize kwargs unchanged.

### Test updates

- `test/test_types.jl:73` — VUMPS default-value assertion:
  ```julia
  @test v.inner_checkpoint === Plain()
  @test v.eig_checkpoint   === Plain()
  @test v.step_checkpoint  === Plain()
  ```
  (was `@test v.ifcheckpoint == false`)
- `test/test_types.jl:108` — QRCTM assertion unchanged

### New tests

New file `test/test_checkpoint.jl`, registered in `test/runtests.jl`:

1. Symbol normalization: `_ckpt_method(:plain) === Plain()`, `_ckpt_method(:recompute) === Recompute()`, `_ckpt_method(:offload) === Offload()`, unknown symbol throws `ArgumentError`
2. `Base.convert` path: `VUMPS(General(); step_checkpoint=:offload).step_checkpoint === Offload()`
3. Inner-map `Offload` rejection: `leftenv(...; alg=VUMPS(General(); inner_checkpoint=Offload()))` throws `ArgumentError`
4. **Gradient equivalence** (critical): small VUMPS run (D=2, χ=4) with identical seeds:
   - `Plain()` vs `Recompute()` at step level: `|∂E_Plain - ∂E_Recompute| / |∂E_Plain| < 1e-9`
   - `Plain()` vs `Offload()` at step level: same tolerance (on CPU, `Offload()` is equivalent to `Recompute()` modulo a copy roundtrip)
5. Combination coverage: `(Plain, Plain, Recompute)`, `(Recompute, Plain, Plain)`, `(Plain, Offload, Offload)` — verify no crash and gradient agreement

## Risks and rollback

- **Namespace risk from `Base.convert`**: defining `convert(::Type{CheckpointMethod}, ::Symbol)` is scoped to our abstract type — cannot conflict with existing Base methods.
- **Adjoint tuple shape**: `@adjoint checkpoint(m::Method, f, args...)` returns `(nothing, ∂f, ∂args...)` — standard Zygote convention for singleton-dispatched args.
- **Inner assert cost**: `_assert_inner_method` is a type-dispatched no-op for `Plain`/`Recompute`; Julia will inline it. Zero runtime cost on the hot path.
- **Rollback**: single-PR revert restores old state. Scope is contained: new file, struct field changes, call-site edits, example kwargs, two test files.

## Non-goals

- Extending inner-map to `Offload` (memory says eig-level offload is already marginal due to StructArray pointer sharing — inner-map would be even finer and yet more marginal).
- Refactoring `QRCTM.ifcheckpoint` or `iPEPSOptimize.ifcheckpoint` (separate semantics, no demand).
- Implementing `Tape()` method (future extension; infrastructure would allow adding it as a new subtype + two methods without touching existing code).
