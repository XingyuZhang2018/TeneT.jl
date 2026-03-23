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

function update!(env::CTMEnv, env′::CTMEnv)
    env.C .= env′.C
    env.T .= env′.T
    return env
end
