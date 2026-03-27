# iPEPS initialization and environment setup
# Handles random initialization, file loading, bond-dimension enlargement,
# and VUMPS runtime environment creation.

# --------------------------------------------------------------------------- #
#  init_ipeps  —  create or load iPEPS site tensors
# --------------------------------------------------------------------------- #

"""
    init_ipeps(; atype=Array, etype=ComplexF64, No, pattern, χ, D, d, params)

Initialize iPEPS tensors.  When `No != 0`, loads from a previously saved
checkpoint file; otherwise creates a random initial state.

# Arguments
- `atype`: array backend (e.g. `Array`, `CuArray`)
- `etype`: element type (e.g. `ComplexF64`, `Float64`)
- `No`:    checkpoint number; `0` means random initialization
- `pattern`: unit-cell pattern matrix (`Matrix{Int}`)
- `χ`:     boundary bond dimension (used for file path)
- `D`:     iPEPS bond dimension
- `d`:     physical dimension
- `params`: an `iPEPSOptimize` instance (provides `folder`, `verbosity`, etc.)

# Returns
An array of shape `(D, D, D, D, d, Nsites)` on the requested backend.
"""
function init_ipeps(; atype=Array, etype=Float64, No::Int=0, d::Int, D::Int, χ::Int, params::iPEPSOptimize)
    Ni, Nj = size(params.pattern)
    N = length(unique(params.pattern))
    if No != 0
        file = joinpath(params.folder, "D$(D)", "ipeps", "χ$(χ)", "No.$(No).jld2")
        @info "load ipeps from file: $file"
        A = load(file, "bcipeps")
    else
        lattice = params.model.lattice
        A = _init_random_ipeps(lattice, etype, D, d, N, Ni, Nj)
        A /= norm(A)
        @info "generate random ipeps at $(joinpath(params.folder, "D$(D)"))"
    end
    return atype(A)
end

# Lattice-dependent random tensor shape
_init_random_ipeps(::Square, etype, D, d, N, Ni, Nj) =
    rand(etype, D, D, D, D, d, N) .+ 1

_init_random_ipeps(::Kagome, etype, D, d, N, Ni, Nj) =
    rand(etype, D, D, D, D, d^3, N) .+ 1

function _init_random_ipeps(::Honeycomb{:merge}, etype, D, d, N, Ni, Nj)
    rand(etype, D, D, D, D, d^2, N) .+ 1
end

function _init_random_ipeps(::Honeycomb{:brickwall}, etype, D, d, N, Ni, Nj)
    Ni % 2 == 0 && Nj % 2 == 0 || throw(ArgumentError("Ni and Nj should be even for brickwall"))
    rand(etype, D, 1, D, D, d, N) .+ 1
end

# --------------------------------------------------------------------------- #
#  init_ipeps_to_D  —  enlarge bond dimension via SU parameterization
# --------------------------------------------------------------------------- #

"""
    init_ipeps_to_D(; atype=Array, No, D, D_new, χ, params)

Load an iPEPS with bond dimension `D` and enlarge it to `D_new` by embedding
into a larger tensor (using SU parameterization for the existing part).
"""
function init_ipeps_to_D(; atype=Array, No, D::Int, D_new::Int, χ::Int, params::iPEPSOptimize)
    file = joinpath(params.folder, "D$(D)", "ipeps", "χ$(χ)", "No.$(No).jld2")
    A = load(file, "bcipeps")
    D_old, d = size(A)[[1, 5]]
    params.verbosity >= 2 && @info "load ipeps from $file"

    A = build_A(A, params)
    A = SU_parameterization(A, params; D_new)

    Nsites = length(unique(params.pattern))
    A_new = rand(eltype(A), D_new, D_new, D_new, D_new, d, Nsites)
    for i in 1:Nsites
        A_new[:,:,:,:,:,i] = A[i][1:D_new, 1:D_new, 1:D_new, 1:D_new, :]
    end
    params.verbosity >= 2 && @info "enlarged iPEPS to D=$D_new, size=$(size(A_new))"
    return atype(A_new)
end

# --------------------------------------------------------------------------- #
#  init_ipeps_perturbation  —  enlarge D with small random perturbation
# --------------------------------------------------------------------------- #

"""
    init_ipeps_perturbation(; atype=Array, No, D, D_new, χ, ϵ=1e-1, params)

Load an iPEPS with bond dimension `D` and embed it into a `D_new`-dimensional
tensor, filling the new entries with a small random perturbation of magnitude
`ϵ * norm(A)`.
"""
function init_ipeps_perturbation(; atype=Array, No, D::Int, D_new::Int, χ::Int, ϵ=1e-1, params::iPEPSOptimize)
    file = joinpath(params.folder, "D$(D)", "ipeps", "χ$(χ)", "No.$(No).jld2")
    A = load(file, "bcipeps")
    D_old, d = size(A)[[1, 5]]
    params.verbosity >= 2 && @info "load ipeps from $file"

    Nsites = length(unique(params.pattern))
    A_new = (rand(eltype(A), D_new, D_new, D_new, D_new, d, Nsites) .- 0.5) * norm(A) * ϵ
    for i in 1:Nsites
        A_new[1:D_old, 1:D_old, 1:D_old, 1:D_old, :, i] = A[:,:,:,:,:,i]
    end
    params.verbosity >= 2 && @info "perturbed iPEPS to D=$D_new, size=$(size(A_new))"
    return atype(A_new)
end

# --------------------------------------------------------------------------- #
#  init_ipeps_from_small_D  —  simple embedding (ADC4PEPS style, single tensor)
# --------------------------------------------------------------------------- #

"""
    init_ipeps_from_small_D(; atype=Array, No, D, D_new, d, χ, ϵ=1e-3, params)

Load a single-site iPEPS tensor of bond dimension `D` and embed into `D_new`.
Used when the iPEPS has no unit-cell pattern (single tensor of shape `D^4 x d`).
"""
function init_ipeps_from_small_D(; atype=Array, No, D::Int, D_new::Int, d::Int, χ::Int, ϵ::Float64=1e-3, params)
    D < D_new || throw(ArgumentError("D=$D should be smaller than D_new=$D_new"))
    file = joinpath(params.folder, "D$(D)", "ipeps", "χ$(χ)", "No.$(No).jld2")
    A = load(file, "bcipeps")
    A_new = ϵ * rand(Float64, D_new, D_new, D_new, D_new, d)
    A_new[1:D, 1:D, 1:D, 1:D, :] = A
    params.verbosity >= 2 && @info "load ipeps from $file, enlarged to D=$D_new"
    return atype(A_new)
end

# --------------------------------------------------------------------------- #
#  initialize_env  —  create or load VUMPS boundary environment
# --------------------------------------------------------------------------- #

"""
    initialize_env(A, D, χ, params::iPEPSOptimize; restriction_ipeps=identity)

Create or load the VUMPS runtime environment for boundary contraction.
If `params.ifload_env` is true and a saved environment exists on disk, it is
loaded; otherwise a fresh `VUMPSRuntime` is constructed from the current
iPEPS tensors.

# Arguments
- `A`: raw iPEPS parameter array
- `D`: bond dimension
- `χ`: boundary bond dimension
- `params`: optimization parameters
- `restriction_ipeps`: optional function that enforces symmetry constraints on `A`
"""
function initialize_env(A, D::Int, χ::Int, params::iPEPSOptimize; restriction_ipeps=identity)
    folder_path = joinpath(params.folder, "D$(D)", "environment")
    file_path = joinpath(folder_path, "χ$χ.jld2")

    if hasproperty(params, :ifload_env) && params.ifload_env
        if ispath(file_path)
            try
                return load_rt(folder_path, _arraytype(A), params.boundary_alg.ifparallelupdown; file="χ$χ.jld2")
            catch e
                @warn "Failed to load environment from $file_path: $(sprint(showerror, e)). Creating new environment."
                return _create_new_env(A, χ, params; restriction_ipeps)
            end
        else
            params.verbosity >= 2 && @warn "File $file_path not found. Creating new VUMPS environment."
            return _create_new_env(A, χ, params; restriction_ipeps)
        end
    else
        return _create_new_env(A, χ, params; restriction_ipeps)
    end
end

"""
    _create_new_env(A, χ, params; restriction_ipeps=identity)

Internal helper: build a fresh `VUMPSRuntime` from the iPEPS tensors.
"""
function _create_new_env(A, χ::Int, params::iPEPSOptimize; restriction_ipeps=identity)
    A = restriction_ipeps(A)
    A = build_A(A, params)
    return init_env(A, χ, params.boundary_alg)
end
