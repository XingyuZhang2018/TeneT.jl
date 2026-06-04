# Boundary environment structs

"""
    VUMPSRuntime

Runtime environment for the VUMPS algorithm, holding the canonical tensors
and left/right fixed-point environments.

# Fields
- `AL`: Left-canonical tensor (StructArray over unit cell)
- `AR`: Right-canonical tensor (StructArray over unit cell)
- `C`:  Center matrix (StructArray over unit cell)
- `FL`: Left fixed-point environment (StructArray over unit cell)
- `FR`: Right fixed-point environment (StructArray over unit cell)
"""
struct VUMPSRuntime
    AL::StructArray
    AR::StructArray
    C::StructArray
    FL::StructArray
    FR::StructArray
end

"""
    VUMPSEnv

Observation environment constructed from up and down VUMPS runtimes,
used for computing expectation values.

# Fields
- `ACu`: Up mixed-canonical tensor
- `ARu`: Up right-canonical tensor
- `ACd`: Down mixed-canonical tensor
- `ARd`: Down right-canonical tensor
- `FLu`: Up left fixed-point environment
- `FRu`: Up right fixed-point environment
- `FLo`: Mixed (observation) left environment
- `FRo`: Mixed (observation) right environment
"""
struct VUMPSEnv
    ACu::StructArray
    ARu::StructArray
    ACd::StructArray
    ARd::StructArray
    FLu::StructArray
    FRu::StructArray
    FLo::StructArray
    FRo::StructArray
end

"""
    PlaquetteVUMPSRuntime

Runtime for the Plaquette VUMPS mode.  Only left-canonical form is used;
there is no AR or FR.

# Fields
- `AL`: Left-canonical MPS tensor
- `C`:  Gauge matrix
- `FL`: Left fixed-point environment
"""
struct PlaquetteVUMPSRuntime
    AL::StructArray
    C::StructArray
    FL::StructArray
end

"""
    PlaquetteVUMPSEnv

Observation environment for Plaquette VUMPS.  Uses two left environments
(upper `FLu` and observation `FLo`) instead of left/right pairs.

# Fields
- `AL`: Left-canonical MPS tensor
- `C`:  Gauge matrix
- `FLu`: Upper left environment (= FL from runtime)
- `FLo`: Observation left environment
"""
struct PlaquetteVUMPSEnv
    AL::StructArray
    C::StructArray
    FLu::StructArray
    FLo::StructArray
end

"""
    OnesideVUMPSEnv

Observation environment for `VUMPS{<:Oneside}` mode. Has both left and right
environments (no L-R symmetry to exploit) but only the up canonical tensors
(no ACd/ARd because U-D hermiticity makes them equal to AC/AR under the
model's `obs_index` row mapping).

# Fields
- `AC`:  Mixed-canonical tensor (= ACu; ACd derived via `obs_index`)
- `AR`:  Right-canonical tensor (= ARu)
- `FLu`: Up left environment (= FL from runtime, ifobs=false)
- `FRu`: Up right environment (= FR from runtime, ifobs=false)
- `FLo`: Observation left environment (from `leftenv_oneside`)
- `FRo`: Observation right environment (from `rightenv_oneside`)
"""
struct OnesideVUMPSEnv
    AC::StructArray
    AR::StructArray
    FLu::StructArray
    FRu::StructArray
    FLo::StructArray
    FRo::StructArray
end

struct C4vVUMPSEnv{CT <: AbstractArray{<:Number, 2}, ET <: Union{leg3, leg4}}
    AL::ET
    C::CT
    FL::ET
end

"""
    CTMEnv{CT, ET}

Corner Transfer Matrix environment for a single site, holding a corner
matrix `C` and an edge tensor `T`.

# Fields
- `C`: Corner matrix (`AbstractArray{<:Number, 2}`)
- `T`: Edge tensor (rank-3 or rank-4 array)
"""
struct CTMEnv{CT<:AbstractArray{<:Number,2}, ET<:Union{AbstractArray{<:Number,3}, AbstractArray{<:Number,4}}}
    C::CT
    T::ET
end

"""
    C3vCTMEnv{CT, RT}

Native honeycomb C3v CTM environment for a single site. `C` is the corner
matrix and `R` is the three-leg edge tensor reshaped as `(χ, D, D, χ)`.
"""
struct C3vCTMEnv{CT<:AbstractArray{<:Number,2}, RT<:AbstractArray{<:Number,4}}
    C::CT
    R::RT
end

"""
    C3vTwoSiteCTMEnv{CT, RT}

Bipartite honeycomb C3v CTM environment. `(CA, RA)` and `(CB, RB)` are the
two alternating A→B and B→A boundary sectors.
"""
struct C3vTwoSiteCTMEnv{CT<:AbstractArray{<:Number,2}, RT<:AbstractArray{<:Number,4}}
    CA::CT
    RA::RT
    CB::CT
    RB::RT
end

# ── GPU/CPU array conversions ────────────────────────────────────────
Array(rt::VUMPSRuntime)    = VUMPSRuntime(Array(rt.AL), Array(rt.AR), Array(rt.C), Array(rt.FL), Array(rt.FR))
CuArray(rt::VUMPSRuntime)  = VUMPSRuntime(CuArray(rt.AL), CuArray(rt.AR), CuArray(rt.C), CuArray(rt.FL), CuArray(rt.FR))
ROCArray(rt::VUMPSRuntime) = VUMPSRuntime(ROCArray(rt.AL), ROCArray(rt.AR), ROCArray(rt.C), ROCArray(rt.FL), ROCArray(rt.FR))
Array(rt::Tuple{VUMPSRuntime,VUMPSRuntime})    = Array.(rt)
CuArray(rt::Tuple{VUMPSRuntime,VUMPSRuntime})  = CuArray.(rt)
ROCArray(rt::Tuple{VUMPSRuntime,VUMPSRuntime}) = ROCArray.(rt)

Array(rt::PlaquetteVUMPSRuntime)    = PlaquetteVUMPSRuntime(Array(rt.AL), Array(rt.C), Array(rt.FL))
CuArray(rt::PlaquetteVUMPSRuntime)  = PlaquetteVUMPSRuntime(CuArray(rt.AL), CuArray(rt.C), CuArray(rt.FL))
ROCArray(rt::PlaquetteVUMPSRuntime) = PlaquetteVUMPSRuntime(ROCArray(rt.AL), ROCArray(rt.C), ROCArray(rt.FL))

Array(env::OnesideVUMPSEnv) = OnesideVUMPSEnv(Array(env.AC), Array(env.AR),
                                              Array(env.FLu), Array(env.FRu),
                                              Array(env.FLo), Array(env.FRo))
CuArray(env::OnesideVUMPSEnv) = OnesideVUMPSEnv(CuArray(env.AC), CuArray(env.AR),
                                                CuArray(env.FLu), CuArray(env.FRu),
                                                CuArray(env.FLo), CuArray(env.FRo))
ROCArray(env::OnesideVUMPSEnv) = OnesideVUMPSEnv(ROCArray(env.AC), ROCArray(env.AR),
                                                 ROCArray(env.FLu), ROCArray(env.FRu),
                                                 ROCArray(env.FLo), ROCArray(env.FRo))

Array(rt::C4vVUMPSEnv)    = C4vVUMPSEnv(Array(rt.AL), Array(rt.C), Array(rt.FL))
CuArray(rt::C4vVUMPSEnv)  = C4vVUMPSEnv(CuArray(rt.AL), CuArray(rt.C), CuArray(rt.FL))
ROCArray(rt::C4vVUMPSEnv) = C4vVUMPSEnv(ROCArray(rt.AL), ROCArray(rt.C), ROCArray(rt.FL))

Array(rt::CTMEnv)    = CTMEnv(Array(rt.C), Array(rt.T))
CuArray(rt::CTMEnv)  = CTMEnv(CuArray(rt.C), CuArray(rt.T))
ROCArray(rt::CTMEnv) = CTMEnv(ROCArray(rt.C), ROCArray(rt.T))

Array(rt::C3vCTMEnv)    = C3vCTMEnv(Array(rt.C), Array(rt.R))
CuArray(rt::C3vCTMEnv)  = C3vCTMEnv(CuArray(rt.C), CuArray(rt.R))
ROCArray(rt::C3vCTMEnv) = C3vCTMEnv(ROCArray(rt.C), ROCArray(rt.R))

Array(rt::C3vTwoSiteCTMEnv) =
    C3vTwoSiteCTMEnv(Array(rt.CA), Array(rt.RA), Array(rt.CB), Array(rt.RB))
CuArray(rt::C3vTwoSiteCTMEnv) =
    C3vTwoSiteCTMEnv(CuArray(rt.CA), CuArray(rt.RA), CuArray(rt.CB), CuArray(rt.RB))
ROCArray(rt::C3vTwoSiteCTMEnv) =
    C3vTwoSiteCTMEnv(ROCArray(rt.CA), ROCArray(rt.RA), ROCArray(rt.CB), ROCArray(rt.RB))

# ── Host-offload methods for checkpoint(Offload(), ...) ───────────────────
# Specialisations that let `checkpoint(Offload(), ...)` walk StructArray and
# VUMPSRuntime args. Base methods live in `src/utils/checkpoint.jl`.

# _atype_of: detect the on-device atype by peeking at a leaf array.
_atype_of(S::StructArray) = isempty(S.data) ? nothing : _atype_of(S.data[1])
_atype_of(rt::VUMPSRuntime) = _atype_of(rt.AL)
_atype_of(rt::PlaquetteVUMPSRuntime) = _atype_of(rt.AL)
_atype_of(env::OnesideVUMPSEnv) = _atype_of(env.AC)
_atype_of(rt::C4vVUMPSEnv) = _atype_of(rt.AL)
_atype_of(rt::C3vCTMEnv) = _atype_of(rt.C)
_atype_of(rt::C3vTwoSiteCTMEnv) = _atype_of(rt.CA)

# _offload_to_host: walk struct, replace each device leaf with a CPU copy.
_offload_to_host(S::StructArray) = StructArray(map(_offload_to_host, S.data), S.pattern)
_offload_to_host(rt::VUMPSRuntime) =
    VUMPSRuntime(_offload_to_host(rt.AL), _offload_to_host(rt.AR),
                 _offload_to_host(rt.C),  _offload_to_host(rt.FL), _offload_to_host(rt.FR))
_offload_to_host(rt::PlaquetteVUMPSRuntime) =
    PlaquetteVUMPSRuntime(_offload_to_host(rt.AL), _offload_to_host(rt.C), _offload_to_host(rt.FL))
_offload_to_host(env::OnesideVUMPSEnv) = OnesideVUMPSEnv(
    _offload_to_host(env.AC), _offload_to_host(env.AR),
    _offload_to_host(env.FLu), _offload_to_host(env.FRu),
    _offload_to_host(env.FLo), _offload_to_host(env.FRo))
_offload_to_host(rt::C4vVUMPSEnv) =
    C4vVUMPSEnv(_offload_to_host(rt.AL), _offload_to_host(rt.C), _offload_to_host(rt.FL))
_offload_to_host(rt::C3vCTMEnv) =
    C3vCTMEnv(_offload_to_host(rt.C), _offload_to_host(rt.R))
_offload_to_host(rt::C3vTwoSiteCTMEnv) =
    C3vTwoSiteCTMEnv(_offload_to_host(rt.CA), _offload_to_host(rt.RA),
                     _offload_to_host(rt.CB), _offload_to_host(rt.RB))

# _to_atype: rebuild an on-device copy from the CPU snapshot using the
# detected atype. Takes no `ref` to the original, so the pullback closure
# of `checkpoint(Offload(), ...)` need not pin the device-side args alive.
_to_atype(atype, S::StructArray) = StructArray(map(d -> _to_atype(atype, d), S.data), S.pattern)
_to_atype(atype, rt::VUMPSRuntime) =
    VUMPSRuntime(_to_atype(atype, rt.AL), _to_atype(atype, rt.AR),
                 _to_atype(atype, rt.C),  _to_atype(atype, rt.FL), _to_atype(atype, rt.FR))
_to_atype(atype, rt::PlaquetteVUMPSRuntime) =
    PlaquetteVUMPSRuntime(_to_atype(atype, rt.AL), _to_atype(atype, rt.C), _to_atype(atype, rt.FL))
_to_atype(atype, env::OnesideVUMPSEnv) = OnesideVUMPSEnv(
    _to_atype(atype, env.AC), _to_atype(atype, env.AR),
    _to_atype(atype, env.FLu), _to_atype(atype, env.FRu),
    _to_atype(atype, env.FLo), _to_atype(atype, env.FRo))
_to_atype(atype, rt::C4vVUMPSEnv) =
    C4vVUMPSEnv(_to_atype(atype, rt.AL), _to_atype(atype, rt.C), _to_atype(atype, rt.FL))
_to_atype(atype, rt::C3vCTMEnv) =
    C3vCTMEnv(_to_atype(atype, rt.C), _to_atype(atype, rt.R))
_to_atype(atype, rt::C3vTwoSiteCTMEnv) =
    C3vTwoSiteCTMEnv(_to_atype(atype, rt.CA), _to_atype(atype, rt.RA),
                     _to_atype(atype, rt.CB), _to_atype(atype, rt.RB))

# ── In-place update helpers ──────────────────────────────────────────
function update!(env::VUMPSRuntime, env′::VUMPSRuntime)
    env.AL.data .= env′.AL.data
    env.AR.data .= env′.AR.data
    env.C.data  .= env′.C.data
    env.FL.data .= env′.FL.data
    env.FR.data .= env′.FR.data
    return env
end

function update!(env::Tuple{VUMPSRuntime,VUMPSRuntime},
                 env′::Tuple{VUMPSRuntime,VUMPSRuntime})
    update!(env[1], env′[1])
    update!(env[2], env′[2])
    return env
end

function update!(env::VUMPSRuntime,
                 env′::Tuple{VUMPSRuntime,VUMPSRuntime})
    update!(env, env′[1])
    return env
end

function update!(env::PlaquetteVUMPSRuntime, env′::PlaquetteVUMPSRuntime)
    env.AL.data .= env′.AL.data
    env.C.data  .= env′.C.data
    env.FL.data .= env′.FL.data
    return env
end

function update!(env::C4vVUMPSEnv, env´::C4vVUMPSEnv) 
    env.AL .= env´.AL
    env.C .= env´.C
    env.FL .= env´.FL
    return env
end

function update!(env::CTMEnv, env′::CTMEnv)
    env.C .= env′.C
    env.T .= env′.T
    return env
end

function update!(env::C3vCTMEnv, env′::C3vCTMEnv)
    env.C .= env′.C
    env.R .= env′.R
    return env
end

function update!(env::C3vTwoSiteCTMEnv, env′::C3vTwoSiteCTMEnv)
    env.CA .= env′.CA
    env.RA .= env′.RA
    env.CB .= env′.CB
    env.RB .= env′.RB
    return env
end
